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
    /// Effective areal heat capacity (J·m⁻²·K⁻¹); larger = more thermal inertia / lag.
    pub heat_capacity: f32,
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
    /// overturning cells, m·s⁻¹. Smaller than the zonal speed, as on Earth.
    pub meridional_wind_speed: f32,
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
    /// Extra lateral heat conductance representing mid-latitude baroclinic-eddy heat transport,
    /// added to `diffusivity` and peaking in the storm-track band. This — not surface advection —
    /// is what carries heat poleward and warms the poles. Same units as `diffusivity`.
    pub eddy_diffusivity: f32,

    // --- Atmosphere layer (two-layer grey radiation + convection) ---
    /// Atmospheric IR emissivity/absorptivity ε_a (0..1): the greenhouse strength. The atmosphere
    /// absorbs ε_a of the surface's thermal radiation and re-emits up and down, warming the surface
    /// above its bare-rock emission temperature. ~0.78 lifts a 255 K surface to ~288 K.
    pub atm_emissivity: f32,
    /// Atmospheric areal heat capacity C_a (J·m⁻²·K⁻¹) — the column thermal mass (~1e7). Its
    /// inertia, coupled to the surface by convection + back-radiation, buffers the diurnal swing.
    pub atm_heat_capacity: f32,
    /// Convective (sensible-heat) exchange coefficient (W·m⁻²·K⁻¹): the upward heat flux per degree
    /// the surface is warmer than the air *beyond the lapse threshold*. This is the vertical venting
    /// — hot surface air carries heat up into the atmosphere instead of the surface running away.
    pub convection_coeff: f32,
    /// Convective lapse threshold Γ (K): convection only fires when `T_s − T_a > Γ`, so the
    /// atmosphere settles ~Γ below the surface. This keeps the air *colder* than the surface, which
    /// is what makes the greenhouse work — without it, convection equalizes the layers and the
    /// greenhouse vanishes. Stands in for the dry/moist adiabatic lapse rate over the layer depth.
    pub convection_threshold: f32,
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
            // ~0.75-day thermal time constant: strong, clearly visible diurnal swing with a
            // realistic afternoon-warm / pre-dawn-cold lag. Areal heat capacity, J·m⁻²·K⁻¹.
            heat_capacity: 4.0e5,
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
            // Storm-track eddy conductance — the dominant meridional heat transport (Budyko–Sellers
            // EBM closure for unresolved baroclinic-eddy flux). Strong enough that down-gradient
            // transport clearly overcomes the equatorward surface advection and warms the poles;
            // self-limiting because it scales with the (shrinking) gradient. Sized against the
            // diffusion CFL (see MAX_SUBSTEP).
            eddy_diffusivity: 150.0,

            // Greenhouse strength. The single-slab value for an Earth-like 288 K mean is ε_a≈0.78,
            // but the strong spatial+diurnal variance here depresses the area-mean (⟨T⁴⟩>⟨T⟩⁴), so
            // a stronger ε_a compensates to keep the mean near Earth's.
            atm_emissivity: 0.90,
            // Atmospheric column heat capacity (~p/g·c_p), much larger than the bare-rock surface.
            atm_heat_capacity: 1.0e7,
            // Sensible-heat exchange coefficient (W·m⁻²·K⁻¹): strong coupling pins the thin surface
            // toward the high-inertia atmosphere (offset by the lapse threshold), capping the daytime
            // overshoot and shrinking the diurnal swing.
            convection_coeff: 35.0,
            // Lapse threshold (K): the atmosphere sits ~33 K below the surface, preserving the
            // greenhouse while convection still caps daytime overshoot.
            convection_threshold: 33.0,
            // The atmosphere mixes heat laterally more freely than the surface.
            atm_diffusivity: 60.0,
        }
    }
}

/// Lowest temperature a cell is allowed to reach (cosmic microwave background), keeping the
/// T⁴ term well-behaved through long nights.
const TEMP_FLOOR: f32 = 2.7;

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
    /// Current per-cell atmosphere-layer temperatures (K).
    temp_a: Vec<f32>,
    /// Scratch buffer for the double-buffered atmosphere update.
    temp_a_scratch: Vec<f32>,
    /// Current per-cell surface wind (tangent to the sphere, m·s⁻¹). Prognostic: integrated from the
    /// momentum balance each step. Seeded from the diagnostic balance on the first [`Sim::advance`].
    wind: Vec<Vec3>,
    /// Scratch buffer for the double-buffered wind update.
    wind_scratch: Vec<Vec3>,
    /// Current per-cell temperature gradient (tangent, K·m⁻¹), recomputed each step and shared by
    /// the wind's pressure-gradient force.
    grad: Vec<Vec3>,
    /// Whether [`wind`] has been seeded from the diagnostic balance yet.
    wind_seeded: bool,
}

impl Sim {
    /// Create a simulation with all cells at a uniform starting temperature. The terrain starts
    /// empty; call [`Sim::generate_terrain`] (which needs the grid) to populate it.
    pub fn new(n: usize, climate: Climate, initial_temp: f32) -> Self {
        Self {
            climate,
            time: 0.0,
            terrain: Terrain::empty(),
            crust: Crust::empty(),
            temp: vec![initial_temp; n],
            scratch: vec![initial_temp; n],
            // Atmosphere starts a bit cooler than the surface (it sits above it).
            temp_a: vec![initial_temp - 20.0; n],
            temp_a_scratch: vec![initial_temp - 20.0; n],
            wind: vec![Vec3::ZERO; n],
            wind_scratch: vec![Vec3::ZERO; n],
            grad: vec![Vec3::ZERO; n],
            wind_seeded: false,
        }
    }

    /// Generate the tectonic plates and the elevation field they imply over `grid` (deterministic
    /// in `seed`). The crust is derived from the plates, so it is generated in the same pass.
    pub fn generate_terrain(&mut self, grid: &Grid, num_plates: usize, seed: u64) {
        self.terrain = Terrain::generate(grid, num_plates, seed);
        self.crust = Crust::generate(grid, &self.terrain, seed);
    }

    /// Current per-cell surface temperatures (K).
    pub fn temperatures(&self) -> &[f32] {
        &self.temp
    }

    /// Current per-cell atmosphere-layer temperatures (K).
    pub fn atmosphere_temperatures(&self) -> &[f32] {
        &self.temp_a
    }

    /// Per-cell surface elevation (m relative to sea level; negative = ocean floor).
    pub fn elevations(&self) -> &[f32] {
        &self.crust.elevation
    }

    /// Current per-cell surface wind (tangent to the sphere, m·s⁻¹). All zero until the first
    /// [`Sim::advance`] has run.
    pub fn winds(&self) -> &[Vec3] {
        &self.wind
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

            // The single equator-to-pole gradient factor scales both the belts and the storm-track
            // eddy diffusivity; compute it (and the temperature gradient that drives the wind's
            // pressure force) once and share.
            let gradient = wind::gradient_factor(grid, &self.temp);
            self.grad = wind::gradients(grid, &self.temp, self.climate.planet_radius);

            // Seed the prognostic wind from the diagnostic balance the first time, so it starts in a
            // sensible state rather than from rest.
            if !self.wind_seeded {
                wind::compute(grid, &self.climate, gradient, &self.grad, &mut self.wind);
                self.wind_seeded = true;
            }

            // Integrate the wind one step (momentum balance: advection + Coriolis + pressure +
            // belt relaxation + viscosity), then swap it in.
            wind::step(
                grid,
                &self.climate,
                gradient,
                &self.grad,
                dt,
                &self.wind,
                &mut self.wind_scratch,
            );
            std::mem::swap(&mut self.wind, &mut self.wind_scratch);

            temperature::step(
                grid,
                &self.climate,
                sun,
                dt,
                TEMP_FLOOR,
                gradient,
                &self.temp,
                &self.temp_a,
                &self.wind,
                &self.grad,
                &mut self.scratch,
                &mut self.temp_a_scratch,
            );
            std::mem::swap(&mut self.temp, &mut self.scratch);
            std::mem::swap(&mut self.temp_a, &mut self.temp_a_scratch);
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
        // Disable the atmosphere too (greenhouse + convection) so the surface reaches the bare
        // σT⁴ = absorbed equilibrium this test checks.
        climate.atm_emissivity = 0.0;
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
}
