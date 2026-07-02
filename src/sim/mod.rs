//! Climate simulation: the day/night cycle and surface temperature.
//!
//! The planet is the unit sphere with +Z as the north pole. Rather than rotating the mesh,
//! we move the **subsolar point** (the direction to the sun) through the planet's fixed
//! frame: its longitude sweeps once per day, and its latitude (solar declination) oscillates
//! once per year with the axial tilt. Each cell then exchanges energy by absorbing sunlight,
//! radiating heat (Stefan–Boltzmann), and conducting laterally to its neighbors.
//!
//! The per-cell update is data-parallel (rayon), reading a stable snapshot and writing a
//! scratch buffer (double buffering), so neighbor reads never race writes.

pub mod elevation;
mod temperature;
pub mod terrain;
pub mod wind;

use glam::Vec3;

use crate::grid::Grid;
use elevation::Crust;
use terrain::Terrain;

/// Stefan–Boltzmann constant (W·m⁻²·K⁻⁴).
pub const STEFAN_BOLTZMANN: f32 = 5.670_374_4e-8;

/// Tunable climate parameters. Defaults are roughly Earth-like and produce a stable
/// day/night thermal lag with an equator-warm / pole-cold gradient.
#[derive(Clone, Copy, Debug)]
pub struct Climate {
    /// Peak solar flux at the subsolar point (W·m⁻²).
    pub solar_constant: f32,
    /// Fraction of incoming sunlight reflected (0..1).
    pub albedo: f32,
    /// Thermal emissivity for outgoing radiation (0..1).
    pub emissivity: f32,
    /// Areal heat capacity of **land** surface (J·m⁻²·K⁻¹) — a thin thermally-active skin, so it
    /// responds fast (a clear diurnal + seasonal swing, as continents have). Applied to cells with
    /// elevation ≥ 0; ocean cells use [`heat_capacity_ocean`] instead.
    pub heat_capacity: f32,
    /// Areal heat capacity of **ocean** surface (J·m⁻²·K⁻¹) — the wind-mixed layer (~70 m of water),
    /// ~500× land. Its huge inertia keeps the sea-surface temperature nearly flat through the day and
    /// the seasons, which (oceans being ~70% of the surface) is what holds each hemisphere's mean
    /// temperature steady instead of swinging ±50 °C with the season. Applied to cells with elevation < 0.
    pub heat_capacity_ocean: f32,
    /// Lateral heat-conduction coefficient (W·m⁻²·K⁻¹ per unit conductance).
    pub diffusivity: f32,
    /// Axial tilt (radians); drives seasonal variation of the subsolar latitude.
    pub obliquity: f32,
    /// Length of one day in simulation seconds.
    pub day_seconds: f32,
    /// Length of one year in simulation seconds.
    pub year_seconds: f32,

    // --- Atmospheric transport (winds) ---
    /// Planet radius (m). Turns the unit-sphere arc lengths into physical distances, needed for
    /// temperature gradients (K·m⁻¹) and so wind speeds in m·s⁻¹.
    pub planet_radius: f32,
    /// Nominal peak speed of the zonal-mean belts (trades / westerlies / polar easterlies), m·s⁻¹,
    /// at the reference equator-to-pole temperature contrast. Scales with the simulated contrast.
    pub trade_wind_speed: f32,
    /// Nominal peak speed of the meridional (poleward/equatorward) surface branch of the
    /// overturning cells, m·s⁻¹. Smaller than the zonal speed, as on Earth. Reused (with reversed
    /// sign) for the upper layer's overturning return branch — mass continuity makes the two
    /// branches comparable in a single column.
    pub meridional_wind_speed: f32,
    /// Nominal peak speed of the upper-tropospheric **subtropical jet** (m·s⁻¹) at the reference
    /// gradient — the westerly maximum at ~30° aloft. Earth's runs ~25–40 m·s⁻¹.
    pub jet_wind_speed: f32,
    /// Pressure-gradient acceleration per unit temperature gradient (m²·s⁻²·K⁻¹) — i.e.
    /// `(dp/dT)/ρ` — converting the temperature gradient into the pressure-gradient force that
    /// drives the geostrophic perturbation (the monsoon / continental-high departures, and the
    /// only time-varying part of the wind as the day/night warm spot moves).
    pub geostrophic_coeff: f32,
    /// Linear surface-friction (Ekman drag) coefficient (s⁻¹) used in the diagnostic *seed* wind.
    /// Turns the near-surface wind across the isobars toward low pressure; keeps the balance finite
    /// at the equator. 0 = pure geostrophic (flow along isobars, singular at the equator).
    pub wind_friction: f32,
    /// Relaxation rate (s⁻¹) pulling the prognostic wind toward the analytic tri-cell belts. This
    /// is what *maintains* the Hadley/Ferrel/Polar structure (which the dynamics can't generate on
    /// their own) and also acts as the surface drag. Larger = belts dominate; smaller = freer eddies.
    pub wind_relax: f32,
    /// Eddy viscosity (s⁻¹, applied to the conductance-weighted velocity Laplacian) damping the
    /// smallest scales so the central momentum advection stays numerically stable.
    pub wind_viscosity: f32,
    /// Rayleigh drag (s⁻¹) on the **upper** wind layer. Much weaker than the surface
    /// [`wind_friction`] — the free troposphere is far from the surface boundary layer — so the
    /// subtropical jet can run fast. Still nonzero for eddy dissipation and the equatorial bound.
    pub wind_friction_upper: f32,
    /// Vertical momentum coupling (s⁻¹): rate at which the upper wind is relaxed toward the surface
    /// wind, representing vertical mixing / the barotropic component (thermal wind: `u_upper ≈
    /// u_lower + shear`). Without it the upper layer, forced only by the slow upper-temperature
    /// field, is nearly steady; this injects the turbulent surface wind's variability aloft so the
    /// jet is lively. Kept below `wind_relax` so the jet's belt shear still survives.
    pub vertical_wind_coupling: f32,
    /// Extra lateral heat conductance representing mid-latitude baroclinic-eddy heat transport,
    /// added to `diffusivity` and peaking in the storm-track band. This — not surface advection —
    /// is what carries heat poleward and warms the poles. Same units as `diffusivity`.
    pub eddy_diffusivity: f32,

    // --- Atmosphere (a column of `n_layers` grey-radiation layers + a convection chain) ---
    /// Number of atmosphere layers stacked above the surface (≥1; capped at [`MAX_LAYERS`]). The
    /// column's per-layer emissivity, heat capacity and lapse are generated from the *totals* below,
    /// so refining the column (more layers) keeps the surface climate ~invariant — it just resolves
    /// the vertical structure more finely. Fixed at [`Sim::new`]; changing it needs a fresh `Sim`.
    pub n_layers: usize,
    /// Total column IR emissivity `Σ ε_l` — the greenhouse strength. Split across the layers by
    /// [`atm_emissivity_decay`]; holding the sum fixed keeps the greenhouse ~independent of `n_layers`.
    pub atm_emissivity_total: f32,
    /// Geometric ratio `ε_{l+1}/ε_l` (<1) of per-layer emissivity with height: less absorber aloft,
    /// so upper layers are optically thinner. At `n_layers = 2` with the default total this yields
    /// the old hand-tuned (0.72, 0.50).
    pub atm_emissivity_decay: f32,
    /// Total column areal heat capacity `Σ C_l` (J·m⁻²·K⁻¹) — the column thermal mass, split across
    /// the layers by [`atm_heat_capacity_decay`].
    pub atm_heat_capacity_total: f32,
    /// Geometric ratio `C_{l+1}/C_l` (<1) of per-layer heat capacity with height (more mass low). At
    /// `n_layers = 2` with the default total this yields the old (6e6, 4e6).
    pub atm_heat_capacity_decay: f32,
    /// Convective (sensible-heat) exchange coefficient (W·m⁻²·K⁻¹) of the **bottom** link
    /// (surface→layer-0, the boundary layer): the upward heat flux per degree of super-threshold
    /// warmth. Hot air carries heat up the column instead of the surface running away.
    pub convection_coeff: f32,
    /// Geometric decay (≤1) of the convective coefficient per link with height. `1.0` couples the
    /// whole column as tightly as the boundary layer (the old uniform behavior); `<1` loosens the
    /// free troposphere so upper layers keep a warm anomaly long enough to travel with the wind —
    /// the key knob for emergent heatwaves — while the boundary-layer link stays strong.
    pub convection_decay: f32,
    /// Total convective lapse (K) from the surface to the top layer. The per-link threshold Γ is this
    /// divided by the number of links (`n_layers`), so a link fires only when the body below is
    /// warmer than the one above by more than Γ. Holding the *total* fixed keeps the overall vertical
    /// structure ~independent of `n_layers` (each link just spans a smaller slice).
    pub convection_lapse: f32,
    /// Ablation switch (diagnostics): when false, the air layers are not advected by their winds.
    /// Always true in production; lets a diagnostic isolate the advection channel.
    pub atm_advection: bool,
    /// Lateral heat conductance within the atmosphere (same units as `diffusivity`); the atmosphere
    /// mixes heat horizontally more freely than the surface.
    pub atm_diffusivity: f32,
}

impl Default for Climate {
    fn default() -> Self {
        Self {
            // Earth's solar constant (total solar irradiance at 1 AU), W·m⁻².
            solar_constant: 1361.0,
            // Earth's mean Bond albedo (dimensionless).
            albedo: 0.3,
            // Surface now radiates as a near-blackbody; the greenhouse is modelled explicitly by the
            // atmosphere layer (`atm_emissivity`), so the old 0.61 fudge is gone. Dimensionless.
            emissivity: 1.0,
            // Land: ~0.75-day thermal time constant — clearly visible diurnal swing with a realistic
            // afternoon-warm / pre-dawn-cold lag. Areal heat capacity, J·m⁻²·K⁻¹.
            heat_capacity: 4.0e5,
            // Ocean mixed layer (~70 m of water): ρ·c_p·depth ≈ 1025·4000·70. ~700× land, so the sea
            // surface barely changes through day or season — damps the hemispheric seasonal swing.
            heat_capacity_ocean: 2.9e8,
            // Lateral heat-conduction coefficient, W·m⁻²·K⁻¹ per unit conductance.
            diffusivity: 25.0,
            // Earth's axial tilt, radians.
            obliquity: 23.44_f32.to_radians(),
            // Earth's solar day, seconds.
            day_seconds: 86_400.0,
            // Earth's year (365.25 days), seconds.
            year_seconds: 86_400.0 * 365.25,

            // Earth's mean radius, metres.
            planet_radius: 6.371e6,
            // Peak surface trade/westerly speed (~7–10 m·s⁻¹ on Earth).
            trade_wind_speed: 9.0,
            // Meridional surface branch is weaker than the zonal flow.
            meridional_wind_speed: 3.0,
            // Subtropical jet peak (~30 m·s⁻¹), the upper-level westerly maximum at 30°.
            jet_wind_speed: 30.0,
            // (dp/dT)/ρ ≈ (67 Pa·K⁻¹)/(1.2 kg·m⁻³): gives a few-m·s⁻¹ geostrophic perturbation for
            // typical gradients, so weather-scale flow is visible alongside the steady belts.
            geostrophic_coeff: 60.0,
            // Light Rayleigh drag (~1.4-day timescale): dissipates eddies and bounds the wind near
            // the equator, without flattening the belts (which the zonal-mean relaxation maintains).
            wind_friction: 8.0e-6,
            // Zonal-mean relaxation (~9-hour timescale): strongly holds the band-mean wind to the
            // tri-cell belts. Acts only on the mean, so it preserves the cells without damping eddies.
            wind_relax: 3.0e-5,
            // Eddy viscosity sized against the momentum CFL (ν·maxΣw·dt well under 1) — small, so
            // turbulence survives down toward the grid scale.
            wind_viscosity: 2.0e-4,
            // Upper-layer Rayleigh drag, equal to the surface value (the free troposphere is
            // weakly damped). This lets the upper eddies turbulence freely — lively upper-wind
            // arrows — which the *two*-layer thermodynamics can afford: with a cold upper layer
            // advected separately from the lower, the overturning transports heat physically rather
            // than over-mixing, so the equator-pole gradient stays Earth-like without heavy upper
            // drag (which a single slab had needed, at the cost of a static jet).
            wind_friction_upper: 8.0e-6,
            // Vertical momentum coupling (~19-hour timescale): below `wind_relax` so the jet's belt
            // shear survives, but strong enough to drag the turbulent surface variability aloft —
            // without it the upper layer (forced only by the slow upper temperature) is nearly steady.
            vertical_wind_coupling: 1.5e-5,
            // Storm-track eddy conductance — the dominant meridional heat transport (Budyko–Sellers
            // EBM closure for unresolved baroclinic-eddy flux). Strong enough that down-gradient
            // transport clearly overcomes the equatorward surface advection and warms the poles;
            // self-limiting because it scales with the (shrinking) gradient. Sized against the
            // diffusion CFL (see MAX_SUBSTEP). NOTE: this also *stabilizes* the surface advection's
            // central-difference form, so it cannot be lowered to offset the new upper-layer
            // transport — damp the upper eddies via `wind_friction_upper` instead.
            eddy_diffusivity: 150.0,

            // A three-layer troposphere: enough vertical resolution for a lower free-tropospheric
            // layer (where weather/heat anomalies can travel with the low-level flow) distinct from
            // the surface-coupled boundary layer and the jet aloft.
            n_layers: 3,
            // Total column greenhouse strength `Σ ε_l`, and its geometric decay with height. At
            // n_layers = 2 these reproduce the old hand-tuned (0.72, 0.50); the totals-based form
            // keeps the greenhouse ~constant as the column is refined (Earth-like surface mean).
            atm_emissivity_total: 1.22,
            atm_emissivity_decay: 0.694,
            // Total column thermal mass (≈ the old 1e7 single slab) and its decay with height (more
            // mass low). At n_layers = 2 these reproduce the old (6e6, 4e6).
            atm_heat_capacity_total: 1.0e7,
            atm_heat_capacity_decay: 0.667,
            // Sensible-heat exchange coefficient (W·m⁻²·K⁻¹) of the boundary-layer link. Higher links
            // decay from this by `convection_decay` (set below).
            convection_coeff: 35.0,
            // Loosen the free troposphere aloft (tuned against `heatwave_diagnostics`): the upper
            // links are ~10× weaker than the boundary layer, so a free-tropospheric warm anomaly is
            // only radiatively/loosely coupled and keeps a multi-day memory instead of being pinned
            // to the surface in ~1 day. The boundary-layer link (0) stays at the full coefficient.
            convection_decay: 0.1,
            // Total surface→top lapse (K): split over the links (Γ = 66/n_layers per link), it builds
            // a realistic tropospheric profile while preserving the greenhouse. At n_layers = 2 this
            // is the old 33 K per link (surface → lower −33 → upper −66).
            convection_lapse: 66.0,
            // Lateral atmospheric mixing. Lowered from the old 60 (tuned against `heatwave_diagnostics`):
            // strong lateral diffusion was smearing warm anomalies away in ~1 day, so a much weaker
            // value lets them persist and travel. It still provides the numerical dissipation that keeps
            // the layers' advective central-difference form stable (verified at the production grid);
            // do not push it near zero without switching temperature advection to an upwind scheme.
            atm_diffusivity: 15.0,
            // Production: the air layers are advected. (Ablation switch for diagnostics only.)
            atm_advection: true,
        }
    }
}

/// Lowest temperature a cell is allowed to reach (cosmic microwave background), keeping the
/// T⁴ term well-behaved through long nights.
const TEMP_FLOOR: f32 = 2.7;

/// Hard cap on the number of atmosphere layers. Bounds the fixed-size scratch arrays in the
/// radiation sweep so the hot loop needs no per-cell allocation.
pub(crate) const MAX_LAYERS: usize = 16;

/// Split a column `total` across `n` layers by a geometric profile with ratio `decay` (layer 0 the
/// largest), normalized so the per-layer values sum to `total`. Used to distribute the column
/// emissivity and heat capacity so the *totals* — hence the surface climate — stay ~invariant as the
/// column is refined into more layers.
fn geometric_profile(total: f32, decay: f32, n: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(n);
    let mut w = 1.0f32;
    let mut sum = 0.0f32;
    for _ in 0..n {
        weights.push(w);
        sum += w;
        w *= decay;
    }
    let scale = if sum > 0.0 { total / sum } else { 0.0 };
    weights.iter().map(|w| w * scale).collect()
}

/// Largest internal integration step (sim seconds). The engine sub-steps to this bound so the
/// explicit Euler update stays stable regardless of the wall-clock frame time. Chosen to keep the
/// diffusion update well within its stability limit dt·D·Σw/C < 1 — now sized for the *peak*
/// conductance (base `diffusivity` plus the storm-track `eddy_diffusivity` at the strongest
/// gradient), which lands the worst-case CFL factor around 0.7. See `diffusion_substep_is_stable`.
const MAX_SUBSTEP: f32 = 150.0;

/// Climate simulation state over a [`Grid`].
pub struct Sim {
    pub climate: Climate,
    /// Accumulated simulation time (seconds).
    pub time: f64,
    /// Tectonic plates and their motion. Empty until [`Sim::generate_terrain`] runs; geology and
    /// climate share this struct because they will couple (currents, orographic effects) later.
    pub terrain: Terrain,
    /// Crust + elevation field, derived from the plates. Empty until [`Sim::generate_terrain`].
    pub crust: Crust,
    /// Current per-cell surface temperatures (K).
    temp: Vec<f32>,
    /// Scratch buffer for the double-buffered surface update.
    scratch: Vec<f32>,
    /// Per-cell surface heat capacity (J·m⁻²·K⁻¹): the ocean mixed-layer value where the cell is
    /// below sea level, the land value above. Recomputed whenever the terrain changes; defaults to
    /// all-land before terrain is generated. Its ocean/land contrast is what damps the seasonal
    /// (and diurnal) swing realistically.
    heat_cap: Vec<f32>,
    /// Number of atmosphere layers (fixed from `climate.n_layers` at construction). Sizes every
    /// per-layer buffer below.
    n_layers: usize,
    /// Current per-cell atmosphere-layer temperatures (K), one contiguous field per layer, index 0 =
    /// lowest/warmest … `n_layers−1` = top/coldest.
    atm: Vec<Vec<f32>>,
    /// Scratch buffers for the double-buffered atmosphere update (one per layer).
    atm_scratch: Vec<Vec<f32>>,
    /// Current per-cell wind for each layer (tangent to the sphere, m·s⁻¹). Prognostic: integrated
    /// from the momentum balance each step, seeded from the diagnostic balance on the first
    /// [`Sim::advance`]. Layer 0 is the near-surface flow (advects air layer 0 and is the displayed
    /// "surface wind"); higher layers carry the jet and the overturning's return branches.
    wind_layers: Vec<Vec<Vec3>>,
    /// Scratch buffers for the double-buffered wind update (one per layer).
    wind_scratch: Vec<Vec<Vec3>>,
    /// Current per-cell surface temperature gradient (tangent, K·m⁻¹), recomputed each step; drives
    /// the layer-0 wind's pressure-gradient force (surface pressure ~ surface temperature).
    grad_s: Vec<Vec3>,
    /// Current per-cell temperature gradient of each atmosphere layer (tangent, K·m⁻¹). `grads[l]`
    /// advects air layer `l`, and (for `l ≥ 1`) drives that layer's wind's pressure force.
    grads: Vec<Vec<Vec3>>,
    /// Current per-cell surface→layer-0 convective (sensible-heat) flux H (W·m⁻²) — the upward venting
    /// off the surface, written each step. Diagnostic only (not fed back); the ITCZ/convection signal.
    conv: Vec<f32>,
    /// Whether the prognostic winds have been seeded from the diagnostic balance yet.
    wind_seeded: bool,
}

impl Sim {
    /// Create a simulation with all cells at a uniform starting temperature. The terrain starts
    /// empty; call [`Sim::generate_terrain`] (which needs the grid) to populate it.
    pub fn new(n: usize, climate: Climate, initial_temp: f32) -> Self {
        let layers = climate.n_layers.max(1);
        assert!(
            layers <= MAX_LAYERS,
            "n_layers {layers} exceeds MAX_LAYERS {MAX_LAYERS}",
        );
        // Each atmosphere layer starts one per-link lapse cooler than the one below it — the chain's
        // equilibrium ordering (surface warmest, top coldest).
        let gamma = climate.convection_lapse / layers as f32;
        let atm: Vec<Vec<f32>> = (0..layers)
            .map(|l| vec![initial_temp - (l as f32 + 1.0) * gamma; n])
            .collect();
        Self {
            climate,
            time: 0.0,
            terrain: Terrain::empty(),
            crust: Crust::empty(),
            temp: vec![initial_temp; n],
            scratch: vec![initial_temp; n],
            // No terrain yet → treat the whole surface as land; recomputed in `generate_terrain`.
            heat_cap: vec![climate.heat_capacity; n],
            n_layers: layers,
            atm_scratch: atm.clone(),
            atm,
            wind_layers: vec![vec![Vec3::ZERO; n]; layers],
            wind_scratch: vec![vec![Vec3::ZERO; n]; layers],
            grad_s: vec![Vec3::ZERO; n],
            grads: vec![vec![Vec3::ZERO; n]; layers],
            conv: vec![0.0; n],
            wind_seeded: false,
        }
    }

    /// Generate the tectonic plates and the elevation field they imply over `grid` (deterministic
    /// in `seed`). The crust is derived from the plates, so it is generated in the same pass.
    pub fn generate_terrain(&mut self, grid: &Grid, num_plates: usize, seed: u64) {
        self.terrain = Terrain::generate(grid, num_plates, seed);
        self.crust = Crust::generate(grid, &self.terrain, seed);
        self.recompute_heat_capacity();
    }

    /// Set each cell's surface heat capacity from its elevation: the ocean mixed-layer value below
    /// sea level, the land value above. Called whenever the terrain (hence the land/sea mask)
    /// changes. Ocean cells barely change temperature through the day or season; this is what keeps
    /// the hemispheres from swinging ±50 °C with the season.
    fn recompute_heat_capacity(&mut self) {
        let (land, ocean) = (self.climate.heat_capacity, self.climate.heat_capacity_ocean);
        for (hc, &elev) in self.heat_cap.iter_mut().zip(self.crust.elevation.iter()) {
            *hc = if elev < 0.0 { ocean } else { land };
        }
    }

    /// Current per-cell surface temperatures (K).
    pub fn temperatures(&self) -> &[f32] {
        &self.temp
    }

    /// Number of atmosphere layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Current per-cell temperatures (K) of atmosphere `layer` (0 = lowest … `n_layers−1` = top).
    pub fn atmosphere_temperatures(&self, layer: usize) -> &[f32] {
        &self.atm[layer]
    }

    /// Current per-cell temperatures (K) of the **lowest** atmosphere layer.
    pub fn lower_atmosphere_temperatures(&self) -> &[f32] {
        &self.atm[0]
    }

    /// Current per-cell temperatures (K) of the **top** atmosphere layer.
    pub fn upper_atmosphere_temperatures(&self) -> &[f32] {
        &self.atm[self.n_layers - 1]
    }

    /// Per-cell surface elevation (m relative to sea level; negative = ocean floor).
    pub fn elevations(&self) -> &[f32] {
        &self.crust.elevation
    }

    /// Current per-cell wind (tangent to the sphere, m·s⁻¹) of atmosphere `layer`. All zero until the
    /// first [`Sim::advance`] has run.
    pub fn layer_winds(&self, layer: usize) -> &[Vec3] {
        &self.wind_layers[layer]
    }

    /// Current per-cell **surface** (lowest-layer) wind (tangent, m·s⁻¹). All zero until the first
    /// [`Sim::advance`] has run.
    pub fn winds(&self) -> &[Vec3] {
        &self.wind_layers[0]
    }

    /// Current per-cell **top**-layer wind (tangent, m·s⁻¹): the subtropical jet + overturning return
    /// branch aloft. All zero until the first [`Sim::advance`] has run.
    pub fn winds_hi(&self) -> &[Vec3] {
        &self.wind_layers[self.n_layers - 1]
    }

    /// Current per-cell convective heat flux H (W·m⁻²) — the upward venting into the atmosphere.
    /// Zero until the first [`Sim::advance`]; highlights the convergence/convection zones (ITCZ).
    pub fn convection(&self) -> &[f32] {
        &self.conv
    }

    /// Unit direction toward the sun in the planet's frame, at the given sim time.
    pub fn sun_direction(&self, time: f64) -> Vec3 {
        let c = &self.climate;
        // Subsolar longitude sweeps once per day (planet spins eastward → sun drifts west).
        let lon = -std::f32::consts::TAU * (time / c.day_seconds as f64) as f32;
        // Solar declination oscillates once per year between ±obliquity.
        let decl = c.obliquity * (std::f32::consts::TAU * (time / c.year_seconds as f64) as f32).sin();
        let (s_decl, c_decl) = decl.sin_cos();
        let (s_lon, c_lon) = lon.sin_cos();
        Vec3::new(c_decl * c_lon, c_decl * s_lon, s_decl)
    }

    /// Advance the simulation by `sim_seconds`, sub-stepping for numerical stability.
    pub fn advance(&mut self, grid: &Grid, sim_seconds: f32) {
        if sim_seconds <= 0.0 {
            return;
        }
        let mut remaining = sim_seconds;
        while remaining > 0.0 {
            let dt = remaining.min(MAX_SUBSTEP);
            let sun = self.sun_direction(self.time);

            let radius = self.climate.planet_radius;
            let n = self.n_layers;

            // Per-layer emissivity and heat capacity from the column totals (cheap; the column is
            // shallow). Generating from totals keeps the greenhouse and thermal mass — hence the
            // surface climate — ~invariant as `n_layers` changes.
            let emis = geometric_profile(
                self.climate.atm_emissivity_total,
                self.climate.atm_emissivity_decay,
                n,
            );
            let heat_cap_atm = geometric_profile(
                self.climate.atm_heat_capacity_total,
                self.climate.atm_heat_capacity_decay,
                n,
            );

            // The single equator-to-pole gradient factor scales both the belts and the storm-track
            // eddy diffusivity; compute it once and share. Then a temperature gradient per field:
            // ∇T_s drives the layer-0 wind, and each air layer's ∇T advects it (and drives its wind
            // aloft).
            let gradient = wind::gradient_factor(grid, &self.temp);
            self.grad_s = wind::gradients(grid, &self.temp, radius);
            for l in 0..n {
                self.grads[l] = wind::gradients(grid, &self.atm[l], radius);
            }

            // Seed the prognostic winds from the diagnostic balance the first time, so they start in
            // a sensible state rather than from rest.
            if !self.wind_seeded {
                for l in 0..n {
                    let drive = if l == 0 { &self.grad_s } else { &self.grads[l] };
                    wind::compute(grid, &self.climate, gradient, drive, &mut self.wind_layers[l], l, n);
                }
                self.wind_seeded = true;
            }

            // Integrate each wind layer one step (momentum balance: advection + Coriolis + pressure +
            // belt relaxation + viscosity), bottom-up so each layer couples vertically to the
            // already-updated layer below it — inheriting the turbulent surface variability aloft, so
            // the jet stays lively. Layer 0's pressure force is driven by ∇T_s; layers aloft by their
            // own ∇T.
            for l in 0..n {
                let drive = if l == 0 { &self.grad_s } else { &self.grads[l] };
                let couple = if l == 0 { None } else { Some(&self.wind_layers[l - 1][..]) };
                wind::step(
                    grid,
                    &self.climate,
                    gradient,
                    drive,
                    dt,
                    &self.wind_layers[l],
                    &mut self.wind_scratch[l],
                    l,
                    n,
                    couple,
                );
                std::mem::swap(&mut self.wind_layers[l], &mut self.wind_scratch[l]);
            }

            // Area-mean of each layer's advective tendency −u·∇T, subtracted in the kernel so
            // advection only redistributes heat within the layer — the stand-in for the vertical mass
            // compensation that makes continuum horizontal advection conserve energy (the imposed
            // winds are divergent, so without it a fast layer's global mean would drift).
            let mut adv_corr = vec![0.0f32; n];
            if self.climate.atm_advection {
                for l in 0..n {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for i in 0..grid.n {
                        let a = grid.areas[i] as f64;
                        num += (-self.wind_layers[l][i].dot(self.grads[l][i])) as f64 * a;
                        den += a;
                    }
                    adv_corr[l] = (num / den) as f32;
                }
            }

            temperature::step(
                grid,
                &self.climate,
                sun,
                dt,
                TEMP_FLOOR,
                gradient,
                n,
                &self.temp,
                &self.atm,
                &emis,
                &heat_cap_atm,
                &self.heat_cap,
                &self.wind_layers,
                &self.grads,
                &adv_corr,
                &mut self.scratch,
                &mut self.atm_scratch,
                &mut self.conv,
            );
            std::mem::swap(&mut self.temp, &mut self.scratch);
            std::mem::swap(&mut self.atm, &mut self.atm_scratch);
            self.time += dt as f64;
            remaining -= dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_direction_is_unit_and_sweeps() {
        let sim = Sim::new(10, Climate::default(), 255.0);
        let s0 = sim.sun_direction(0.0);
        assert!((s0.length() - 1.0).abs() < 1e-5);
        // After a quarter day the subsolar longitude has rotated ~90°.
        let s_quarter = sim.sun_direction((sim.climate.day_seconds / 4.0) as f64);
        assert!(s0.dot(s_quarter).abs() < 0.1, "quarter-day sun should be ~orthogonal");
        // After a full day it returns close to the start.
        let s_full = sim.sun_direction(sim.climate.day_seconds as f64);
        assert!(s0.dot(s_full) > 0.999, "full-day sun should return to start");
    }

    #[test]
    fn subsolar_cell_approaches_radiative_equilibrium() {
        // A tiny grid, sun fixed (no daily motion) by making a day extremely long, and no
        // diffusion: the most-lit cell should converge toward its radiative equilibrium.
        let g = Grid::new(600);
        let mut climate = Climate::default();
        climate.diffusivity = 0.0;
        // Disable all lateral transport so the most-lit cell reaches *pure* radiative equilibrium.
        climate.eddy_diffusivity = 0.0;
        climate.trade_wind_speed = 0.0;
        climate.meridional_wind_speed = 0.0;
        climate.geostrophic_coeff = 0.0;
        // Disable the atmosphere too (all grey layers + convection) so the surface reaches the
        // bare σT⁴ = absorbed equilibrium this test checks.
        climate.atm_emissivity_total = 0.0;
        climate.convection_coeff = 0.0;
        climate.day_seconds = 1.0e12; // effectively frozen sun
        climate.year_seconds = 1.0e15;
        let mut sim = Sim::new(g.n, climate, 200.0);
        // Run for a long time to settle.
        for _ in 0..4000 {
            sim.advance(&g, MAX_SUBSTEP);
        }
        let sun = sim.sun_direction(0.0);
        // Find the most-lit cell and its expected equilibrium.
        let (best, _) = g
            .centers
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.dot(sun)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let cos = g.centers[best].dot(sun).max(0.0);
        let absorbed = (1.0 - climate.albedo) * climate.solar_constant * cos;
        let expected = (absorbed / (climate.emissivity * STEFAN_BOLTZMANN)).powf(0.25);
        let got = sim.temperatures()[best];
        assert!(
            (got - expected).abs() / expected < 0.02,
            "subsolar cell {got}K vs equilibrium {expected}K",
        );
    }

    #[test]
    fn diffusion_substep_is_stable() {
        // Explicit-Euler diffusion is stable when dt·D·max(Σⱼwᵢⱼ)/C < 1 (well under 2). The peak
        // conductance is the base diffusivity plus the storm-track eddy term at its strongest:
        // `eddy_diffusivity` × the maximum gradient factor (clamped to 2.0) × the storm-band peak
        // (1.0). Sizing the substep against this worst case keeps every cell stable.
        let g = Grid::new(8000);
        let c = Climate::default();
        let max_wsum = (0..g.n)
            .map(|i| g.neighbor_weights(i).iter().sum::<f32>())
            .fold(0.0f32, f32::max);
        let max_d = c.diffusivity + c.eddy_diffusivity * 2.0;
        let factor = MAX_SUBSTEP * max_d * max_wsum / c.heat_capacity;
        assert!(factor < 1.0, "diffusion CFL factor {factor} (max Σw = {max_wsum}, D = {max_d}) is unstable");
    }

    #[test]
    fn temperatures_stay_finite_and_bounded_over_a_day() {
        let g = Grid::new(1500);
        let mut sim = Sim::new(g.n, Climate::default(), 255.0);
        // Simulate two full days.
        let day = sim.climate.day_seconds;
        for _ in 0..200 {
            sim.advance(&g, day * 2.0 / 200.0);
        }
        for &t in sim.temperatures() {
            assert!(t.is_finite(), "temperature went non-finite");
            assert!((TEMP_FLOOR..600.0).contains(&t), "temperature out of range: {t}");
        }
    }

    /// The column profiles reduce to the old hand-tuned two-layer values, and always sum to the
    /// column total (the invariant that keeps the climate steady as the column is refined).
    #[test]
    fn geometric_profile_reduces_to_old_two_layer_values() {
        let e = geometric_profile(1.22, 0.694, 2);
        assert!((e[0] - 0.72).abs() < 0.01 && (e[1] - 0.50).abs() < 0.01, "emissivity {e:?}");
        let c = geometric_profile(1.0e7, 0.667, 2);
        assert!((c[0] - 6.0e6).abs() < 5e4 && (c[1] - 4.0e6).abs() < 5e4, "heat cap {c:?}");
        // The sum is preserved for any layer count.
        for n in 1..=8 {
            let p = geometric_profile(1.22, 0.7, n);
            assert!((p.iter().sum::<f32>() - 1.22).abs() < 1e-4, "n={n} sum {:?}", p);
        }
    }

    /// Global-mean run helper: advance an `layers`-deep column for `days` and return the mean
    /// surface temperature. No terrain (uniform fast land heat capacity) so it settles quickly.
    fn mean_surface_temp(layers: usize, days: f32, steps: usize) -> f32 {
        let g = Grid::new(400);
        let mut climate = Climate::default();
        climate.n_layers = layers;
        let mut sim = Sim::new(g.n, climate, 260.0);
        let total = climate.day_seconds * days;
        for _ in 0..steps {
            sim.advance(&g, total / steps as f32);
        }
        sim.temperatures().iter().sum::<f32>() / g.n as f32
    }

    /// Because the per-layer emissivity/heat-capacity/lapse are generated from column *totals*, the
    /// global-mean surface climate is nearly independent of how many layers the column is split into
    /// — refining the column adds vertical detail without moving the surface temperature.
    #[test]
    fn surface_climate_is_roughly_layer_count_invariant() {
        let t2 = mean_surface_temp(2, 10.0, 130);
        let t4 = mean_surface_temp(4, 10.0, 130);
        assert!((t2 - t4).abs() < 12.0, "N=2 {t2:.1}K vs N=4 {t4:.1}K differ too much");
        for (n, t) in [(2, t2), (4, t4)] {
            assert!((250.0..320.0).contains(&t), "N={n} surface mean {t:.1}K implausible");
        }
    }

    /// The grey column + convective chain build a monotonic lapse: in the global mean the surface is
    /// warmest and each layer is colder than the one below it (the greenhouse ordering), for the
    /// default three-layer column.
    #[test]
    fn column_builds_a_monotonic_lapse() {
        let g = Grid::new(1200);
        let climate = Climate::default();
        let mut sim = Sim::new(g.n, climate, 260.0);
        let day = climate.day_seconds;
        for _ in 0..150 {
            sim.advance(&g, day * 5.0 / 150.0);
        }
        let mean = |t: &[f32]| t.iter().sum::<f32>() / t.len() as f32;
        let mut prev = mean(sim.temperatures());
        for l in 0..sim.n_layers() {
            let tl = mean(sim.atmosphere_temperatures(l));
            assert!(tl < prev, "layer {l} mean {tl:.1}K not colder than the body below {prev:.1}K");
            prev = tl;
        }
    }

    /// Heatwave-tuning diagnostic (heavy — ignored). For each candidate coupling it reports the
    /// mean-climate health (surface mean, equator-pole gradient, surface→top lapse) and a heatwave
    /// **twin experiment**: two identical spun-up sims, a warm bump injected into the lowest
    /// free-tropospheric layer of one, then the difference field tracked — its peak amplitude
    /// (persistence) and centroid displacement (does it travel, and poleward?). Run with:
    ///   cargo test --release heatwave_diagnostics -- --ignored --nocapture
    #[test]
    #[ignore]
    fn heatwave_diagnostics() {
        use glam::Vec3;
        // 6000 cells: a snappy on-demand grid. Bump to 16384 (the production grid) for a definitive
        // stability check of the fast jet layer under low diffusion.
        let g = Grid::new(6000);
        let day = Climate::default().day_seconds;

        // (label, convection_decay, atm_diffusivity) — lapse/emissivity kept at defaults.
        let configs = [
            ("baseline  d=1.00 atmD=60", 1.00f32, 60.0f32),
            ("tuned     d=0.10 atmD=15", 0.10, 15.0),
        ];

        for (label, decay, atm_d) in configs {
            let mut climate = Climate::default();
            climate.convection_decay = decay;
            climate.atm_diffusivity = atm_d;

            // Two identical sims spun up together so their (deterministic) eddy fields match.
            let mut ctrl = Sim::new(g.n, climate, 260.0);
            let mut pert = Sim::new(g.n, climate, 260.0);
            let (spin_days, spin_steps) = (4.0f32, 80);
            for _ in 0..spin_steps {
                let dt = day * spin_days / spin_steps as f32;
                ctrl.advance(&g, dt);
                pert.advance(&g, dt);
            }

            // Mean-climate health from the control.
            let area: f64 = (0..g.n).map(|i| g.areas[i] as f64).sum();
            let wmean = |f: &[f32]| {
                ((0..g.n).map(|i| f[i] as f64 * g.areas[i] as f64).sum::<f64>() / area) as f32
            };
            let band = |f: &[f32], lo: f32, hi: f32| {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for i in 0..g.n {
                    let lat = g.lonlat_deg[i].y.abs();
                    if lat >= lo && lat < hi {
                        num += f[i] as f64 * g.areas[i] as f64;
                        den += g.areas[i] as f64;
                    }
                }
                (num / den.max(1.0)) as f32
            };
            let mean_s = wmean(ctrl.temperatures());
            let grad = band(ctrl.temperatures(), 0.0, 15.0) - band(ctrl.temperatures(), 75.0, 90.0);
            let lapse_now = mean_s - wmean(ctrl.atmosphere_temperatures(ctrl.n_layers() - 1));

            // Inject a warm bump into the lowest free-tropospheric layer at 45°N, 0°E.
            let free_l = 1usize.min(ctrl.n_layers() - 1);
            let lat0 = 45.0f32.to_radians();
            let n0 = Vec3::new(lat0.cos(), 0.0, lat0.sin());
            let (amp0, r) = (8.0f32, 0.16f32);
            for i in 0..g.n {
                let ang = g.centers[i].dot(n0).clamp(-1.0, 1.0).acos();
                pert.atm[free_l][i] += amp0 * (-(ang / r) * (ang / r)).exp();
            }

            let smax = ctrl.temperatures().iter().cloned().fold(f32::MIN, f32::max);
            let smin = ctrl.temperatures().iter().cloned().fold(f32::MAX, f32::min);
            println!("\n=== {label} ===");
            println!("  mean-climate: surface {mean_s:6.1}K   eq-pole grad {grad:5.1}K   surf-top lapse {lapse_now:5.1}K   surf-range [{smin:.0},{smax:.0}]K");
            print!("  layer ranges (checkerboard check):");
            for l in 0..ctrl.n_layers() {
                let t = ctrl.atmosphere_temperatures(l);
                let mx = t.iter().cloned().fold(f32::MIN, f32::max);
                let mn = t.iter().cloned().fold(f32::MAX, f32::min);
                print!("  L{l}[{mn:.0},{mx:.0}]");
            }
            println!();
            println!("  heatwave (layer {free_l}):  day    amp(K)   travel(km)   centroid-lat");

            let (run_days, run_steps, every) = (7.0f32, 140, 20);
            let radius_km = climate.planet_radius / 1000.0;
            for step in 0..=run_steps {
                if step % every == 0 {
                    let (mut amp, mut acc, mut wsum) = (0.0f32, Vec3::ZERO, 0.0f32);
                    for i in 0..g.n {
                        let d = pert.atm[free_l][i] - ctrl.atm[free_l][i];
                        if d > amp {
                            amp = d;
                        }
                        if d > 0.0 {
                            acc += g.centers[i] * (d * g.areas[i]);
                            wsum += d * g.areas[i];
                        }
                    }
                    let (travel, clat) = if wsum > 0.0 {
                        let c = (acc / wsum).normalize_or_zero();
                        (c.dot(n0).clamp(-1.0, 1.0).acos() * radius_km, c.z.clamp(-1.0, 1.0).asin().to_degrees())
                    } else {
                        (0.0, 0.0)
                    };
                    let dday = step as f32 * run_days / run_steps as f32;
                    println!("                          {dday:5.1}   {amp:6.2}   {travel:9.0}   {clat:9.1}");
                }
                if step < run_steps {
                    let dt = day * run_days / run_steps as f32;
                    ctrl.advance(&g, dt);
                    pert.advance(&g, dt);
                }
            }
        }
    }
}
