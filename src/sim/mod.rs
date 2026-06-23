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
}

impl Default for Climate {
    fn default() -> Self {
        Self {
            // Earth's solar constant (total solar irradiance at 1 AU), W·m⁻².
            solar_constant: 1361.0,
            // Earth's mean Bond albedo (dimensionless).
            albedo: 0.3,
            // Effective emissivity < 1 to crudely stand in for greenhouse warming: it lifts the
            // global-mean radiative equilibrium from the bare-rock 255 K up to Earth's ~288 K
            // (εσT⁴ = S(1−α)/4 ⇒ T ≈ 288 K). Dimensionless.
            emissivity: 0.61,
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
        }
    }
}

/// Lowest temperature a cell is allowed to reach (cosmic microwave background), keeping the
/// T⁴ term well-behaved through long nights.
const TEMP_FLOOR: f32 = 2.7;

/// Largest internal integration step (sim seconds). The engine sub-steps to this bound so the
/// explicit Euler update stays stable regardless of the wall-clock frame time. Chosen to keep
/// the diffusion update well within its stability limit dt·D·Σw/C < 1 (here ≈ 0.26).
const MAX_SUBSTEP: f32 = 700.0;

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
    /// Current per-cell temperatures (K).
    temp: Vec<f32>,
    /// Scratch buffer for the double-buffered update.
    scratch: Vec<f32>,
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
        }
    }

    /// Generate the tectonic plates and the elevation field they imply over `grid` (deterministic
    /// in `seed`). The crust is derived from the plates, so it is generated in the same pass.
    pub fn generate_terrain(&mut self, grid: &Grid, num_plates: usize, seed: u64) {
        self.terrain = Terrain::generate(grid, num_plates, seed);
        self.crust = Crust::generate(grid, &self.terrain, seed);
    }

    /// Current per-cell temperatures (K).
    pub fn temperatures(&self) -> &[f32] {
        &self.temp
    }

    /// Per-cell surface elevation (m relative to sea level; negative = ocean floor).
    pub fn elevations(&self) -> &[f32] {
        &self.crust.elevation
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
            temperature::step(
                grid,
                &self.climate,
                sun,
                dt,
                TEMP_FLOOR,
                &self.temp,
                &mut self.scratch,
            );
            std::mem::swap(&mut self.temp, &mut self.scratch);
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
        // Explicit-Euler diffusion is stable when dt·D·max(Σⱼwᵢⱼ)/C < 1 (well under 2).
        let g = Grid::new(8000);
        let c = Climate::default();
        let max_wsum = (0..g.n)
            .map(|i| g.neighbor_weights(i).iter().sum::<f32>())
            .fold(0.0f32, f32::max);
        let factor = MAX_SUBSTEP * c.diffusivity * max_wsum / c.heat_capacity;
        assert!(factor < 1.0, "diffusion CFL factor {factor} (max Σw = {max_wsum}) is unstable");
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
