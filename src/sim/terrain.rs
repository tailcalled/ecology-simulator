//! Continental plates: a partition of the grid into rigid tectonic plates, each carrying a
//! direction of motion. This is the first geological layer; later it will feed elevation and
//! couple back into the climate (ocean currents, orographic effects), so it lives alongside the
//! climate state in [`Sim`](super::Sim) rather than in a wholly separate world.
//!
//! Generation is deterministic from a seed. Plates are grown by a randomized flood fill over the
//! grid adjacency graph — dropping a seed cell per plate, then repeatedly claiming a random
//! frontier cell for its plate — which yields organic, interlocking plate outlines rather than
//! the convex cells a nearest-seed Voronoi assignment would give.
//!
//! Each plate carries a heavy-tailed (Pareto) growth weight, and growth picks a plate with
//! probability proportional to its weight before claiming one of *its* frontier cells. This makes
//! a plate's growth rate — and so its final area — track its weight rather than its perimeter,
//! reproducing Earth's strongly unequal, power-law-like plate-size distribution (a few giant
//! plates plus a long tail of small ones). Uniform growth, where rate scales with perimeter,
//! instead drives every plate toward the same area (a random-Voronoi partition).
//!
//! Each plate moves as a rigid rotation of the sphere about its own **Euler pole** (an axis
//! through the planet's center), which is the physically correct way a rigid cap moves on a
//! sphere. A cell's surface velocity is then `ω × p`, automatically tangent to the surface and
//! vanishing at the pole — so a single plate's arrows fan out and swirl around its pole.

use glam::Vec3;

use crate::grid::Grid;

/// One rigid plate: a rotation of the sphere about `axis` (a unit Euler pole through the center)
/// at signed angular speed `rate` (radians per simulation second; sign sets the spin direction).
#[derive(Clone, Copy, Debug)]
pub struct Plate {
    /// Euler pole: unit axis the plate rotates about.
    pub axis: Vec3,
    /// Signed angular speed about `axis` (rad·s⁻¹).
    pub rate: f32,
}

impl Plate {
    /// Surface velocity this plate imparts at unit position `p` (tangent to the sphere).
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        (self.axis * self.rate).cross(p)
    }
}

/// How plate growth-rate weights are drawn, which sets the shape of the plate-size distribution.
/// A plate's growth rate (hence its final area) is proportional to its weight, so the weight
/// distribution maps fairly directly onto the area distribution (modulo geometry).
#[derive(Clone, Copy, Debug)]
pub enum SizeModel {
    /// Pareto weights `u^(−1/exponent)`: a true power-law tail (matches the Sornette/Bird view of
    /// real plates). Smaller exponent → heavier tail → more unequal. Produces extreme giants.
    Pareto(f32),
    /// Lognormal weights `exp(sigma·z)`, `z ~ N(0, 1)`: a lighter extreme tail than Pareto at the
    /// same overall inequality. Larger sigma → more unequal.
    Lognormal(f32),
}

impl SizeModel {
    fn draw(self, rng: &mut Rng) -> f32 {
        match self {
            SizeModel::Pareto(exponent) => rng.pareto(exponent),
            SizeModel::Lognormal(sigma) => rng.lognormal(sigma),
        }
    }
}

/// The tectonic partition of a [`Grid`]: which plate each cell belongs to, the plates' motions,
/// and the per-cell surface velocity those motions produce.
pub struct Terrain {
    /// Plate index per cell (parallel to the grid's cells).
    pub plate_id: Vec<u32>,
    /// The plates, indexed by the ids in `plate_id`.
    pub plates: Vec<Plate>,
    /// Per-cell tectonic surface velocity (`ω × p`), tangent to the sphere.
    pub velocity: Vec<Vec3>,
}

impl Terrain {
    /// An empty terrain (no plates), used before generation.
    pub fn empty() -> Self {
        Self { plate_id: Vec::new(), plates: Vec::new(), velocity: Vec::new() }
    }

    /// Partition `grid` into `num_plates` plates by randomized weighted flood fill, assign each
    /// plate a random Euler pole and speed, and precompute the per-cell surface velocity.
    /// Deterministic in `seed`. Uses the default size model tuned to Earth's distribution.
    pub fn generate(grid: &Grid, num_plates: usize, seed: u64) -> Self {
        Self::generate_with(grid, num_plates, seed, DEFAULT_SIZE_MODEL)
    }

    /// As [`Terrain::generate`], but with an explicit plate-size [`SizeModel`]. Exposed for tuning
    /// the generated size distribution against real plate areas.
    pub fn generate_with(grid: &Grid, num_plates: usize, seed: u64, model: SizeModel) -> Self {
        let n = grid.n;
        let num_plates = num_plates.clamp(1, n.max(1));
        let mut rng = Rng::new(seed);

        const UNASSIGNED: u32 = u32::MAX;
        let mut plate_id = vec![UNASSIGNED; n];
        // Each plate keeps its own frontier of unassigned cells adjacent to it, so we can grow a
        // chosen plate by exactly one cell per step (rate set by weight, not perimeter).
        let mut frontier: Vec<Vec<u32>> = vec![Vec::new(); num_plates];

        // --- Plant one distinct seed cell per plate. ---
        let mut planted = 0usize;
        while planted < num_plates {
            let c = (rng.next_u64() % n as u64) as usize;
            if plate_id[c] == UNASSIGNED {
                plate_id[c] = planted as u32;
                for &nb in grid.neighbors(c) {
                    if plate_id[nb as usize] == UNASSIGNED {
                        frontier[planted].push(nb);
                    }
                }
                planted += 1;
            }
        }

        // --- Heavy-tailed growth weight per plate; the model sets the size distribution's shape. ---
        let weights: Vec<f32> = (0..num_plates).map(|_| model.draw(&mut rng)).collect();

        // --- Grow until every cell is claimed. Each step: pick a still-growing plate with
        // probability ∝ its weight, then claim one random cell from its frontier. ---
        loop {
            // Total weight over plates that can still grow (non-empty frontier).
            let total: f32 = (0..num_plates)
                .filter(|&p| !frontier[p].is_empty())
                .map(|p| weights[p])
                .sum();
            if total <= 0.0 {
                break; // no plate can grow → every reachable cell is claimed
            }

            // Weighted pick among the growing plates.
            let mut target = rng.unit_f32() * total;
            let mut chosen = usize::MAX;
            for p in 0..num_plates {
                if frontier[p].is_empty() {
                    continue;
                }
                target -= weights[p];
                if target <= 0.0 {
                    chosen = p;
                    break;
                }
            }
            if chosen == usize::MAX {
                // Floating-point slack: fall back to the last plate that can still grow.
                chosen = (0..num_plates).rev().find(|&p| !frontier[p].is_empty()).unwrap();
            }

            // Claim one random cell from the chosen plate, discarding stale frontier entries
            // (cells another plate already took). If it drains dry, the outer loop re-picks.
            while !frontier[chosen].is_empty() {
                let k = (rng.next_u64() % frontier[chosen].len() as u64) as usize;
                let cell = frontier[chosen].swap_remove(k) as usize;
                if plate_id[cell] != UNASSIGNED {
                    continue;
                }
                plate_id[cell] = chosen as u32;
                for &nb in grid.neighbors(cell) {
                    if plate_id[nb as usize] == UNASSIGNED {
                        frontier[chosen].push(nb);
                    }
                }
                break;
            }
        }
        // The grid is connected, so every cell is claimed; guard the rare stray anyway.
        for id in plate_id.iter_mut() {
            if *id == UNASSIGNED {
                *id = 0;
            }
        }

        // --- Give each plate a random Euler pole and a signed angular speed. ---
        let plates: Vec<Plate> = (0..num_plates)
            .map(|_| {
                let axis = rng.unit_vec();
                // Magnitude in [0.3, 1.0] of the base rate, with a random spin direction. The
                // absolute value is arbitrary for now (only relative arrow lengths matter); it is
                // expressed as rad·s⁻¹ so it can drive real plate displacement later.
                let mag = 0.3 + 0.7 * rng.unit_f32();
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                Plate { axis, rate: BASE_RATE * mag * sign }
            })
            .collect();

        // --- Precompute the per-cell surface velocity from its plate's motion. ---
        let velocity: Vec<Vec3> = (0..n)
            .map(|i| plates[plate_id[i] as usize].velocity_at(grid.centers[i]))
            .collect();

        Self { plate_id, plates, velocity }
    }
}

/// Base plate angular speed (rad·s⁻¹). Earth's fast plates drift ~10 cm·yr⁻¹ ≈ 5e-17 rad·s⁻¹;
/// this stand-in is far larger because nothing integrates it yet — it only sets arrow scale.
const BASE_RATE: f32 = 1.0;

/// Upper clamp on a plate's growth weight, purely a guard against f32 overflow / one plate
/// swallowing the sphere under a pathological draw. Set high enough that it never triggers at the
/// default σ (exp(0.7·z) reaches it only for z > ~13), so the production distribution is unaffected.
const MAX_WEIGHT: f32 = 1.0e4;

/// Default plate-size model (see `examples/plate_diagnostics.rs`). At the ~12-plate scale the
/// simulator uses, a lognormal fits Earth's PB2002 (Bird 2003) plates on Gini, CV, *and* top-plate
/// share at once — reproducing the real "plateau of ~7 large plates then a tail" shape — whereas a
/// Gini-matched Pareto over-concentrates the largest plate and over-fills the small-plate tail.
const DEFAULT_SIZE_MODEL: SizeModel = SizeModel::Lognormal(DEFAULT_SIGMA);
/// Lognormal sigma for the default model: fitted so mean CV ≈ 2.14 matches Earth's full 52-plate
/// PB2002 catalogue (the count the sim generates). This also lands top-plate share ≈ 25% (close to
/// Earth's 20%); Gini comes out ≈ 0.73, below Earth's 0.82 because the grid can't resolve Earth's
/// microplate tail (smallest plate ≥ 1 cell). σ is the weight spread and scales with the target
/// CV, so fewer plates would call for a smaller σ (≈0.66 to match just the 12 largest).
const DEFAULT_SIGMA: f32 = 1.56;

/// Tiny deterministic PRNG (SplitMix64). Self-contained so plate generation is reproducible
/// without pulling in a random-number crate, matching the grid's deterministic construction.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    fn unit_f32(&mut self) -> f32 {
        // Top 24 bits → exact float in [0, 1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pareto-distributed weight `u^(−1/exponent)` for `u ∈ (0, 1)`: mostly near 1 with a heavy
    /// tail of large values. Smaller `exponent` → heavier tail. Clamped to tame the rare extreme
    /// draw so a single plate can't swallow nearly the whole sphere.
    fn pareto(&mut self, exponent: f32) -> f32 {
        let u = self.unit_f32().max(1e-6);
        u.powf(-1.0 / exponent.max(0.1)).min(MAX_WEIGHT)
    }

    /// Standard-normal sample via the Box–Muller transform.
    fn normal(&mut self) -> f32 {
        let u1 = self.unit_f32().max(1e-6);
        let u2 = self.unit_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Lognormal weight `exp(sigma·z)`, `z ~ N(0, 1)`. The location parameter is fixed at 0
    /// because only relative weights matter — `exp(mu)` would scale all weights equally and
    /// cancel out of the growth-selection probabilities. Clamped like [`Self::pareto`].
    fn lognormal(&mut self, sigma: f32) -> f32 {
        (sigma * self.normal()).exp().min(MAX_WEIGHT)
    }

    /// Uniformly-distributed unit vector on the sphere (z uniform, azimuth uniform).
    fn unit_vec(&mut self) -> Vec3 {
        let z = self.unit_f32() * 2.0 - 1.0;
        let theta = self.unit_f32() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cell_is_assigned_to_a_valid_plate() {
        let g = Grid::new(2000);
        let t = Terrain::generate(&g, 10, 42);
        assert_eq!(t.plate_id.len(), g.n);
        assert_eq!(t.plates.len(), 10);
        assert!(t.plate_id.iter().all(|&id| (id as usize) < t.plates.len()));
        // Every plate actually claimed at least its seed cell.
        for p in 0..t.plates.len() as u32 {
            assert!(t.plate_id.contains(&p), "plate {p} claimed no cells");
        }
    }

    #[test]
    fn velocity_is_tangent_to_the_surface() {
        let g = Grid::new(2000);
        let t = Terrain::generate(&g, 8, 7);
        for i in 0..g.n {
            // ω × p is perpendicular to p, so the radial component must be ~0.
            assert!(t.velocity[i].dot(g.centers[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let g = Grid::new(1500);
        let a = Terrain::generate(&g, 12, 12345);
        let b = Terrain::generate(&g, 12, 12345);
        assert_eq!(a.plate_id, b.plate_id);
        // Different seed gives a different partition.
        let c = Terrain::generate(&g, 12, 99999);
        assert_ne!(a.plate_id, c.plate_id);
    }
}
