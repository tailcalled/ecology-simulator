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

mod temperature;

use glam::Vec3;

use crate::grid::Grid;

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
            solar_constant: 1361.0,
            albedo: 0.3,
            emissivity: 1.0,
            // ~0.75-day thermal time constant: strong, clearly visible diurnal swing with a
            // realistic afternoon-warm / pre-dawn-cold lag.
            heat_capacity: 4.0e5,
            diffusivity: 25.0,
            obliquity: 23.44_f32.to_radians(),
            day_seconds: 86_400.0,
            year_seconds: 86_400.0 * 365.0,
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
    /// Current per-cell temperatures (K).
    temp: Vec<f32>,
    /// Scratch buffer for the double-buffered update.
    scratch: Vec<f32>,
}

impl Sim {
    /// Create a simulation with all cells at a uniform starting temperature.
    pub fn new(n: usize, climate: Climate, initial_temp: f32) -> Self {
        Self {
            climate,
            time: 0.0,
            temp: vec![initial_temp; n],
            scratch: vec![initial_temp; n],
        }
    }

    /// Current per-cell temperatures (K).
    pub fn temperatures(&self) -> &[f32] {
        &self.temp
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
