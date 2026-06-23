//! Terrain elevation: a static, physically-grounded height field built on top of the tectonic
//! plates ([`Terrain`]). This is the second geological layer; the headline output is a per-cell
//! [`elevation`](Crust::elevation) in metres relative to the emergent sea level, which later feeds
//! back into the climate (orographic effects, ocean vs. land albedo / heat capacity) and is the
//! coarse substrate the higher-resolution river/lake layer will carve.
//!
//! The model is deliberately built from physical quantities rather than fitted directly to a
//! target height histogram:
//!
//! 1. **Crust type.** Continents are ancient and have rifted and re-merged across many plate
//!    configurations, so where continental crust sits does not track *today's* plate boundaries.
//!    We seed it as plate-independent coherent "cratons" from multi-octave noise on the sphere,
//!    calibrated so continental crust covers ~40 % of the surface (Earth's value, shelves
//!    included). A continuous continental fraction `φ ∈ [0,1]` then sets each cell's crust
//!    thickness and density.
//!
//! 2. **Tectonics.** Relative plate motion across a boundary (from the plates' Euler poles) is
//!    split into a convergence and a shear component, classifying every boundary cell as
//!    convergent / divergent / transform. Convergence *thickens* crust (orogeny / arcs);
//!    continental divergence *thins* it (rifts); oceanic divergence builds a thermally-buoyant
//!    mid-ocean ridge; oceanic subduction cuts a flexural trench. Each effect decays away from its
//!    boundary over a physically-sized distance.
//!
//! 3. **Isostasy.** Every cell is a crust column floating on the mantle (Airy isostasy): its
//!    rock-surface height is `t·(1 − ρ_crust/ρ_mantle)`. This alone reproduces Earth's bimodal
//!    continent/ocean-floor split (~4.5 km contrast) before any water.
//!
//! 4. **Flooding.** A *fixed ocean water volume* (Earth's, normalised to the unit sphere) is
//!    poured over the topography; the sea level — and hence the land fraction — *emerges* from the
//!    hypsometry rather than being imposed. Submerged columns ride lower under the water load
//!    (water-loaded isostasy), realistically deepening the basins.
//!
//! Generation is deterministic in the seed, like the grid and the plates.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::Vec3;

use crate::grid::Grid;
use crate::rng::Rng;
use super::terrain::Terrain;

// --- Physical constants (SI; densities kg·m⁻³, lengths as noted) ------------------------------

/// Upper-mantle density the crust floats on.
const RHO_MANTLE: f32 = 3300.0;
/// Continental crust density (felsic, light) — the value at `φ = 1`.
const RHO_CONT: f32 = 2800.0;
/// Oceanic crust density (mafic, dense) — the value at `φ = 0`.
const RHO_OCEAN: f32 = 2900.0;
/// Sea-water density (loads submerged crust down).
const RHO_WATER: f32 = 1030.0;
/// Mean Earth radius, to convert physical distances (km) to angular distances (radians) and
/// Earth's ocean volume (m³) to the unit-sphere measure used here.
const EARTH_RADIUS_M: f32 = 6.371e6;

/// Oceanic crust thickness (km) — the value at `φ = 0`. Earth's is ~7 km.
const T_OCEAN_KM: f32 = 7.0;
/// Floor on crust thickness (km) after rift thinning, so a column never inverts.
const MIN_THICKNESS_KM: f32 = 4.0;
/// `φ` above this counts a cell as "continental" when classifying which side of a convergence
/// subducts and which overrides.
const CONT_THRESH: f32 = 0.5;
/// Standard deviation (km) of the intra-continental thickness noise — the source of highlands and
/// basins *within* continents (scaled by `φ`, so oceans stay smooth) before any tectonics. Tuned
/// so the continental surface-elevation spread (≈ this · 0.15 · 1000 m, plus orogenic tails) lands
/// near Earth's ~0.8 km, giving mean land height ≈ Earth's rather than tall continental tablelands.
const CONT_RELIEF_KM: f32 = 3.5;
/// Width (in units of the unit-variance craton noise) of the smooth land↔ocean crust transition.
/// Larger = broader continental shelves / gentler margins.
const MARGIN_WIDTH: f32 = 0.6;

// --- Tectonic effect magnitudes (the main calibration knobs) ----------------------------------

/// Characteristic plate speed used to normalise convergence/shear rates to ~[0, 1]. The plate
/// generator draws angular rates around `BASE_RATE = 1`, so relative speeds are O(1) in these
/// units (the absolute scale is arbitrary until plate motion is actually integrated).
const REF_SPEED: f32 = 1.0;
/// Normalised |convergence| below this is treated as a transform/inactive margin (no uplift).
const CONV_EPS: f32 = 0.04;
/// Maximum crustal thickening (km) added at a full-rate convergent margin (orogeny / arc). Roughly
/// doubles the crust at a head-on continental collision (Tibet ≈ 35 → 70 km).
const OROG_THICKEN_KM: f32 = 38.0;
/// Maximum crustal thinning (km) at a full-rate continental rift.
const RIFT_THIN_KM: f32 = 14.0;
/// Mid-ocean-ridge crest height (m) above the abyssal plain (thermal buoyancy of young
/// lithosphere). Applied to oceanic crust, decaying with distance ≈ age from the ridge.
const RIDGE_AMP_M: f32 = 3000.0;
/// Extra depth (m) of a subduction trench at the boundary (flexural, non-isostatic).
const TRENCH_DEPTH_M: f32 = 4500.0;

/// Convert a physical distance in km to an angular distance (radians) on the unit sphere.
const fn km_rad(km: f32) -> f32 {
    km / (EARTH_RADIUS_M / 1000.0)
}

/// Orogenic belt half-width (Himalaya/Tibet scale): narrow, so mountains are tall but cover little
/// area — matching Earth's high-peak / low-mean-land hypsometry rather than broad high plateaus.
const L_OROG: f32 = km_rad(150.0);
/// Ridge flank length from crest to abyssal plain (where half-space cooling has flattened out).
const L_RIDGE: f32 = km_rad(1500.0);
/// Trench falloff: narrow, the flexural moat hugs the boundary.
const L_TRENCH: f32 = km_rad(150.0);
/// Continental-rift shoulder width.
const L_RIFT: f32 = km_rad(250.0);

/// Coarse plate-boundary classification per cell (for the debug layer and later coupling). The
/// finer subtype (orogeny vs. arc vs. trench, ridge vs. rift) only affects how elevation is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryKind {
    /// Cell is in a plate interior (no differing-plate neighbour).
    None = 0,
    /// Plates approaching: orogeny, volcanic arc, or trench.
    Convergent = 1,
    /// Plates separating: mid-ocean ridge or continental rift.
    Divergent = 2,
    /// Plates sliding past: little vertical relief.
    Transform = 3,
}

/// Tunable generation parameters. Only a few knobs, each physically meaningful; the defaults are
/// Earth-anchored. Exposed so `examples/terrain_diagnostics.rs` can sweep them against Earth's
/// hypsometry.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Ocean water budget in unit-sphere measure (metres·steradian): `Σ depth·area`. The sea
    /// level and land fraction emerge from flooding the topography with exactly this much water.
    pub water_volume: f32,
    /// Target fraction of surface *area* underlain by continental crust (shelves included).
    pub continental_area: f32,
    /// Mean continental crust thickness (km) at `φ = 1`, before tectonics.
    pub mean_cont_thickness_km: f32,
}

/// Earth's ocean volume (1.335×10¹⁸ m³) normalised by R² into the unit-sphere `depth·area`
/// measure, so `water_volume` is a genuine physical quantity rather than a fudge factor.
pub const EARTH_WATER_VOLUME: f32 = 1.335e18 / (EARTH_RADIUS_M * EARTH_RADIUS_M);

impl Default for Params {
    fn default() -> Self {
        Self {
            water_volume: EARTH_WATER_VOLUME,
            continental_area: 0.40,
            mean_cont_thickness_km: 37.0,
        }
    }
}

/// The elevation layer over a [`Grid`]: per-cell crust state plus the flooded height field.
pub struct Crust {
    /// Continental fraction `φ ∈ [0,1]` per cell (0 = pure oceanic crust, 1 = thick continental).
    pub continental_fraction: Vec<f32>,
    /// Crust thickness per cell (km), after tectonic thickening/thinning.
    pub thickness_km: Vec<f32>,
    /// Crust density per cell (kg·m⁻³).
    pub density: Vec<f32>,
    /// Coarse plate-boundary classification per cell.
    pub boundary: Vec<BoundaryKind>,
    /// Surface elevation per cell in metres relative to sea level (negative = ocean floor).
    pub elevation: Vec<f32>,
    /// Emergent sea-surface height on the isostatic (above-mantle-datum) scale, metres.
    pub sea_level: f32,
    /// Emergent fraction of surface area above sea level (a diagnostic, not an input).
    pub land_fraction: f32,
}

impl Crust {
    /// An empty crust (no cells), used before generation.
    pub fn empty() -> Self {
        Self {
            continental_fraction: Vec::new(),
            thickness_km: Vec::new(),
            density: Vec::new(),
            boundary: Vec::new(),
            elevation: Vec::new(),
            sea_level: 0.0,
            land_fraction: 0.0,
        }
    }

    /// Generate the elevation field over `grid` from its plates `terrain`, with default
    /// (Earth-anchored) parameters. Deterministic in `seed`.
    pub fn generate(grid: &Grid, terrain: &Terrain, seed: u64) -> Self {
        Self::generate_with(grid, terrain, seed, Params::default())
    }

    /// As [`Crust::generate`] but with explicit [`Params`], for calibration.
    pub fn generate_with(grid: &Grid, terrain: &Terrain, seed: u64, params: Params) -> Self {
        let n = grid.n;
        if n == 0 {
            return Self::empty();
        }
        let mut rng = Rng::new(seed);

        // --- 1. Craton field: coherent continental blobs from low-frequency sphere noise. ---
        // Low base frequency → a handful of continent-scale features; several octaves give organic
        // coastlines. Plate-independent by construction (sampled in 3D space, not per plate).
        let craton = SphereNoise::new(&mut rng, 5, 14, 1.7, 2.0, 0.55);
        let relief = SphereNoise::new(&mut rng, 4, 12, 5.0, 2.0, 0.5);
        let raw: Vec<f32> = grid.centers.iter().map(|&p| craton.sample(p)).collect();

        // Pick the threshold so the chosen *area* fraction lies above it, then smoothstep across a
        // margin to get a continuous φ (sharp coasts but tapered shelves). Calibrating on area (not
        // a fixed noise value) makes the continental coverage robust to the noise's exact spread.
        let tau = area_quantile(&raw, &grid.areas, params.continental_area);
        let mut continental_fraction = vec![0.0f32; n];
        for i in 0..n {
            continental_fraction[i] = smoothstep(tau - MARGIN_WIDTH, tau + MARGIN_WIDTH, raw[i]);
        }

        // --- 2. Crust thickness & density from φ (+ intra-continental relief). ---
        let mut thickness_km = vec![0.0f32; n];
        let mut density = vec![0.0f32; n];
        for i in 0..n {
            let phi = continental_fraction[i];
            let base = T_OCEAN_KM + phi * (params.mean_cont_thickness_km - T_OCEAN_KM);
            // Relief only inside continents (scaled by φ): highlands and intracratonic basins.
            let bump = phi * CONT_RELIEF_KM * relief.sample(grid.centers[i]);
            thickness_km[i] = (base + bump).max(MIN_THICKNESS_KM);
            density[i] = RHO_OCEAN + phi * (RHO_CONT - RHO_OCEAN);
        }

        // --- 3. Classify plate boundaries and collect per-effect source cells + intensities. ---
        let mut boundary = vec![BoundaryKind::None; n];
        let mut orog_src: Vec<(usize, f32)> = Vec::new(); // crustal thickening (orogeny + arcs)
        let mut rift_src: Vec<(usize, f32)> = Vec::new(); // crustal thinning (continental rift)
        let mut ridge_src: Vec<(usize, f32)> = Vec::new(); // thermal ridge buoyancy (oceanic)
        let mut trench_src: Vec<(usize, f32)> = Vec::new(); // flexural trench (subducting side)

        for i in 0..n {
            let pi = grid.centers[i];
            let vi = terrain.velocity[i];
            let plate_i = terrain.plate_id[i];
            let phi_i = continental_fraction[i];

            // Find this cell's strongest differing-plate neighbour and classify that margin.
            let mut best_speed = 0.0f32;
            for &jn in grid.neighbors(i) {
                let j = jn as usize;
                if terrain.plate_id[j] == plate_i {
                    continue;
                }
                // Outward boundary normal at i: the tangent direction toward neighbour j.
                let d = grid.centers[j] - pi;
                let tangent = d - pi * d.dot(pi);
                let nrm = tangent.normalize_or_zero();
                if nrm == Vec3::ZERO {
                    continue;
                }
                let dv = vi - terrain.velocity[j];
                let conv = dv.dot(nrm); // >0: i advances on j (closing); <0: opening
                let shear = (dv - nrm * conv).length();
                let speed = conv.abs().max(shear);
                if speed <= best_speed {
                    continue;
                }
                best_speed = speed;

                let conv_n = (conv.abs() / REF_SPEED).clamp(0.0, 1.0);
                let phi_j = continental_fraction[j];
                if conv > CONV_EPS * REF_SPEED {
                    boundary[i] = BoundaryKind::Convergent;
                    if phi_i > CONT_THRESH && phi_j > CONT_THRESH {
                        // Continent–continent collision: both sides pile up (orogeny).
                        set_source(&mut orog_src, i, conv_n);
                    } else if phi_i < phi_j {
                        // i is the denser (more oceanic) plate → it subducts → trench on i.
                        set_source(&mut trench_src, i, conv_n);
                    } else {
                        // i overrides the subducting plate → volcanic-arc / cordillera thickening.
                        set_source(&mut orog_src, i, conv_n);
                    }
                } else if conv < -CONV_EPS * REF_SPEED {
                    boundary[i] = BoundaryKind::Divergent;
                    if phi_i < CONT_THRESH {
                        set_source(&mut ridge_src, i, conv_n); // mid-ocean ridge
                    } else {
                        set_source(&mut rift_src, i, conv_n); // continental rift
                    }
                } else {
                    boundary[i] = BoundaryKind::Transform;
                }
            }
        }

        // --- 4. Spread each effect away from its boundary (nearest-source geodesic distance). ---
        let orog = influence(grid, &orog_src, 4.0 * L_OROG);
        let rift = influence(grid, &rift_src, 5.0 * L_RIFT);
        let ridge = influence(grid, &ridge_src, 3.0 * L_RIDGE);
        let trench = influence(grid, &trench_src, 5.0 * L_TRENCH);

        // --- 5. Apply isostatically-compensated effects (thicken/thin crust) → base elevation. ---
        for i in 0..n {
            if orog.dist[i].is_finite() {
                thickness_km[i] += OROG_THICKEN_KM * orog.intensity[i] * (-orog.dist[i] / L_OROG).exp();
            }
            if rift.dist[i].is_finite() {
                thickness_km[i] -= RIFT_THIN_KM * rift.intensity[i] * (-rift.dist[i] / L_RIFT).exp();
            }
            thickness_km[i] = thickness_km[i].max(MIN_THICKNESS_KM);
        }

        // Airy isostasy: rock-surface height above the mantle datum, in metres.
        let mut e_total: Vec<f32> = (0..n)
            .map(|i| thickness_km[i] * 1000.0 * (1.0 - density[i] / RHO_MANTLE))
            .collect();

        // --- 6. Add non-isostatic (thermal / flexural) relief on top of the isostatic base. ---
        for i in 0..n {
            // These act on oceanic crust only; gate by φ so they never lift continents.
            if continental_fraction[i] >= CONT_THRESH {
                continue;
            }
            // Mid-ocean ridge: hot young lithosphere stands high, sinking as depth ∝ √age and age
            // ∝ distance from the ridge (half-space cooling), flattening to the abyssal plain.
            if ridge.dist[i].is_finite() {
                let age = (ridge.dist[i] / L_RIDGE).clamp(0.0, 1.0);
                e_total[i] += RIDGE_AMP_M * ridge.intensity[i] * (1.0 - age.sqrt());
            }
            // Subduction trench: a sharp flexural moat at the boundary (not Airy-compensated).
            if trench.dist[i].is_finite() {
                e_total[i] -= TRENCH_DEPTH_M * trench.intensity[i] * (-trench.dist[i] / L_TRENCH).exp();
            }
        }

        // --- 7. Flood with the fixed water volume → emergent sea level, then final elevation. ---
        let sea_level = solve_sea_level(&e_total, &grid.areas, params.water_volume);
        let k = RHO_WATER / RHO_MANTLE;
        let mut elevation = vec![0.0f32; n];
        let mut land_area = 0.0f64;
        let mut total_area = 0.0f64;
        for i in 0..n {
            let r = rock_surface(e_total[i], sea_level, k);
            elevation[i] = r - sea_level;
            total_area += grid.areas[i] as f64;
            if elevation[i] > 0.0 {
                land_area += grid.areas[i] as f64;
            }
        }
        let land_fraction = (land_area / total_area) as f32;

        Self {
            continental_fraction,
            thickness_km,
            density,
            boundary,
            elevation,
            sea_level,
            land_fraction,
        }
    }
}

/// Water-loaded isostatic rock-surface height of a column whose unloaded (dry) height is `e`, given
/// a sea surface at `s` and water/mantle density ratio `k`. Subaerial columns keep their dry
/// height; submerged columns ride lower because the overlying water column presses them down.
fn rock_surface(e: f32, s: f32, k: f32) -> f32 {
    if e >= s {
        e
    } else {
        (e - k * s) / (1.0 - k)
    }
}

/// Total water volume (unit-sphere `depth·area` measure) held by sea level `s` over topography
/// `e_total`. Monotonically increasing in `s`, which lets us invert it by bisection.
fn water_volume_at(e_total: &[f32], areas: &[f32], s: f32, k: f32) -> f64 {
    let mut v = 0.0f64;
    for i in 0..e_total.len() {
        if e_total[i] < s {
            let r = rock_surface(e_total[i], s, k);
            v += (s - r) as f64 * areas[i] as f64;
        }
    }
    v
}

/// Find the sea level that holds exactly `target` water volume over the topography, by bisection.
fn solve_sea_level(e_total: &[f32], areas: &[f32], target: f32) -> f32 {
    let k = RHO_WATER / RHO_MANTLE;
    let target = target as f64;
    let mut lo = e_total.iter().copied().fold(f32::INFINITY, f32::min);
    let mut hi = e_total.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Raise the upper bracket until it holds at least the target volume (drowned planet limit).
    while water_volume_at(e_total, areas, hi, k) < target && hi < lo + 1.0e6 {
        hi += (hi - lo).max(1000.0);
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if water_volume_at(e_total, areas, mid, k) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Record a boundary source, keeping the strongest intensity if the cell already has one.
fn set_source(sources: &mut Vec<(usize, f32)>, cell: usize, intensity: f32) {
    if intensity <= 0.0 {
        return;
    }
    if let Some(e) = sources.iter_mut().find(|(c, _)| *c == cell) {
        e.1 = e.1.max(intensity);
    } else {
        sources.push((cell, intensity));
    }
}

/// Per-cell result of spreading a boundary effect outward: geodesic distance to the nearest source
/// cell, and that source's intensity. Cells beyond `max_dist` (or with no sources) stay at
/// `dist = ∞`, `intensity = 0`.
struct Influence {
    dist: Vec<f32>,
    intensity: Vec<f32>,
}

/// Multi-source Dijkstra over the grid adjacency: each cell gets the geodesic (great-circle)
/// distance to the nearest source and carries that source's intensity along the shortest path.
/// Bounded by `max_dist` so a sparse boundary doesn't sweep the whole globe.
fn influence(grid: &Grid, sources: &[(usize, f32)], max_dist: f32) -> Influence {
    let n = grid.n;
    let mut dist = vec![f32::INFINITY; n];
    let mut intensity = vec![0.0f32; n];
    let mut heap: BinaryHeap<State> = BinaryHeap::new();
    for &(c, inten) in sources {
        if inten <= 0.0 {
            continue;
        }
        if dist[c] != 0.0 {
            dist[c] = 0.0;
        }
        intensity[c] = intensity[c].max(inten);
    }
    for &(c, _) in sources {
        if dist[c] == 0.0 {
            // Push once per source cell with its (max) intensity. Duplicates are harmless: the
            // `d > dist[cell]` guard skips stale pops.
            heap.push(State { dist: 0.0, cell: c as u32, intensity: intensity[c] });
        }
    }
    while let Some(State { dist: d, cell, intensity: inten }) = heap.pop() {
        let cell = cell as usize;
        if d > dist[cell] {
            continue;
        }
        for &jn in grid.neighbors(cell) {
            let j = jn as usize;
            let nd = d + arc(grid.centers[cell], grid.centers[j]);
            if nd < dist[j] && nd <= max_dist {
                dist[j] = nd;
                intensity[j] = inten;
                heap.push(State { dist: nd, cell: jn, intensity: inten });
            }
        }
    }
    Influence { dist, intensity }
}

/// A min-heap entry for [`influence`], ordered by ascending distance (intensity is carried, not
/// compared). Ties are broken by cell index so the ordering is total and deterministic.
struct State {
    dist: f32,
    cell: u32,
    intensity: f32,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.cell == other.cell
    }
}
impl Eq for State {}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap, so reverse the distance comparison to pop the
        // smallest distance first.
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Great-circle angle (radians) between two unit vectors; robust near 0 and π.
fn arc(a: Vec3, b: Vec3) -> f32 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Hermite smoothstep from 0 (at `edge0`) to 1 (at `edge1`).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The value `v` such that the total `area` of cells with `values[i] > v` is the fraction `frac`
/// of the whole — i.e. the area-weighted upper quantile of `values`.
fn area_quantile(values: &[f32], areas: &[f32], frac: f32) -> f32 {
    let total: f64 = areas.iter().map(|&a| a as f64).sum();
    let target = (1.0 - frac.clamp(0.0, 1.0) as f64) * total; // area to accumulate from the bottom
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut acc = 0.0f64;
    for &i in &idx {
        acc += areas[i] as f64;
        if acc >= target {
            return values[i];
        }
    }
    *idx.last().map(|&i| &values[i]).unwrap_or(&0.0)
}

/// Coherent scalar noise on the unit sphere by spectral synthesis: a sum of plane-wave sinusoids
/// over several octaves. Because each wave is defined in 3D space and sampled on the sphere, the
/// field is seamless — no poles, no antimeridian seam, no grid artefacts — and cheap to evaluate.
/// Output is approximately zero-mean, unit-variance.
struct SphereNoise {
    waves: Vec<Wave>,
    norm: f32,
}

struct Wave {
    dir: Vec3,
    freq: f32,
    phase: f32,
    amp: f32,
}

impl SphereNoise {
    /// Build `octaves` bands of `per_octave` random plane waves. `base_freq` is the lowest band's
    /// angular frequency (sets feature size: wavelength ≈ 2π/freq); `lacunarity` multiplies the
    /// frequency and `persistence` the amplitude between successive octaves.
    fn new(
        rng: &mut Rng,
        octaves: u32,
        per_octave: u32,
        base_freq: f32,
        lacunarity: f32,
        persistence: f32,
    ) -> Self {
        let mut waves = Vec::new();
        let mut power = 0.0f32;
        for o in 0..octaves {
            let freq = base_freq * lacunarity.powi(o as i32);
            let amp = persistence.powi(o as i32);
            for _ in 0..per_octave {
                waves.push(Wave {
                    dir: rng.unit_vec(),
                    freq,
                    phase: rng.unit_f32() * std::f32::consts::TAU,
                    amp,
                });
                power += 0.5 * amp * amp; // variance of amp·sin(·) with uniform phase
            }
        }
        Self { waves, norm: 1.0 / power.sqrt().max(1e-6) }
    }

    /// Sample the field at unit position `p` (≈ N(0, 1)).
    fn sample(&self, p: Vec3) -> f32 {
        let mut s = 0.0f32;
        for w in &self.waves {
            s += w.amp * (w.freq * w.dir.dot(p) + w.phase).sin();
        }
        s * self.norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_world(n: usize, seed: u64) -> (Grid, Terrain, Crust) {
        let g = Grid::new(n);
        let t = Terrain::generate(&g, 12, seed);
        let c = Crust::generate(&g, &t, seed);
        (g, t, c)
    }

    #[test]
    fn generation_is_deterministic() {
        let g = Grid::new(2000);
        let t = Terrain::generate(&g, 12, 7);
        let a = Crust::generate(&g, &t, 99);
        let b = Crust::generate(&g, &t, 99);
        assert_eq!(a.elevation, b.elevation);
        assert_eq!(a.continental_fraction, b.continental_fraction);
        let c = Crust::generate(&g, &t, 100);
        assert_ne!(a.elevation, c.elevation);
    }

    #[test]
    fn elevations_are_finite_and_split_around_sea_level() {
        let (_, _, c) = gen_world(4000, 1);
        assert!(c.elevation.iter().all(|e| e.is_finite()));
        assert!(c.elevation.iter().any(|&e| e > 0.0), "no land");
        assert!(c.elevation.iter().any(|&e| e < 0.0), "no ocean");
        // Sea level is exactly the zero of `elevation` by construction.
        let max_land = c.elevation.iter().cloned().fold(f32::MIN, f32::max);
        let min_sea = c.elevation.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max_land > 0.0 && min_sea < 0.0);
    }

    #[test]
    fn continental_coverage_matches_target() {
        // Area with φ > 0.5 should land near the requested continental_area (±, from noise + the
        // smoothstep margin). Default target is 0.40.
        let g = Grid::new(8000);
        let t = Terrain::generate(&g, 12, 3);
        let c = Crust::generate(&g, &t, 3);
        let total: f64 = g.areas.iter().map(|&a| a as f64).sum();
        let cont: f64 = (0..g.n)
            .filter(|&i| c.continental_fraction[i] > 0.5)
            .map(|i| g.areas[i] as f64)
            .sum();
        let frac = cont / total;
        assert!((0.30..0.50).contains(&frac), "continental area fraction {frac}");
    }

    #[test]
    fn water_volume_is_conserved() {
        let g = Grid::new(4000);
        let t = Terrain::generate(&g, 12, 5);
        let params = Params::default();
        let c = Crust::generate_with(&g, &t, 5, params);
        // The water depth at each ocean cell is exactly −elevation (elevation = rock − sea level),
        // so summing depth·area must recover the budget we poured in.
        let mut v = 0.0f64;
        for i in 0..g.n {
            if c.elevation[i] < 0.0 {
                v += (-c.elevation[i]) as f64 * g.areas[i] as f64;
            }
        }
        let rel = (v - params.water_volume as f64).abs() / params.water_volume as f64;
        assert!(rel < 0.02, "water volume off by {rel} (got {v}, want {})", params.water_volume);
    }

    #[test]
    fn land_fraction_is_plausible() {
        let (_, _, c) = gen_world(8000, 2);
        assert!(
            (0.10..0.55).contains(&c.land_fraction),
            "land fraction {} implausible",
            c.land_fraction
        );
    }

    #[test]
    fn thicker_lighter_crust_stands_higher() {
        // Airy isostasy sanity: among land cells, thickness·(1−ρ/ρm) should track elevation.
        let (_, _, c) = gen_world(4000, 8);
        let mut pairs: Vec<(f32, f32)> = (0..c.elevation.len())
            .map(|i| (c.thickness_km[i] * (1.0 - c.density[i] / RHO_MANTLE), c.elevation[i]))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The lightest/thinnest decile should sit far below the heaviest/thickest decile.
        let lo = &pairs[..pairs.len() / 10];
        let hi = &pairs[pairs.len() * 9 / 10..];
        let lo_mean = lo.iter().map(|p| p.1).sum::<f32>() / lo.len() as f32;
        let hi_mean = hi.iter().map(|p| p.1).sum::<f32>() / hi.len() as f32;
        assert!(hi_mean > lo_mean, "isostasy inverted: {hi_mean} !> {lo_mean}");
    }
}
