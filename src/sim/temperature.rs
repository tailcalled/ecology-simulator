//! The per-cell energy-balance kernel — the hot loop, run in parallel with rayon.
//!
//! For each cell, over one time step `dt`:
//!   absorbed = (1 − albedo) · S · max(0, n·ŝ)      [incoming sunlight]
//!   emitted  = emissivity · σ · T⁴                  [outgoing thermal radiation]
//!   diffuse  = D · Σⱼ wᵢⱼ (Tⱼ − Tᵢ)                 [lateral conduction to neighbors]
//!   T' = T + dt · (absorbed − emitted + diffuse) / C
//!
//! Reads come from `temp` (a stable snapshot); writes go to `out` (scratch). The caller swaps
//! the buffers afterward, so neighbor reads never observe a half-updated step.

use glam::Vec3;
use rayon::prelude::*;

use super::{Climate, STEFAN_BOLTZMANN};
use crate::grid::Grid;

/// Advance every cell one explicit-Euler step. `out` must be the same length as `temp`.
pub fn step(
    grid: &Grid,
    climate: &Climate,
    sun: Vec3,
    dt: f32,
    temp_floor: f32,
    temp: &[f32],
    out: &mut [f32],
) {
    let absorb_coeff = (1.0 - climate.albedo) * climate.solar_constant;
    let emit_coeff = climate.emissivity * STEFAN_BOLTZMANN;
    let inv_c = 1.0 / climate.heat_capacity;
    let diffusivity = climate.diffusivity;

    out.par_iter_mut().enumerate().for_each(|(i, t_out)| {
        let ti = temp[i];

        let cos_incidence = grid.centers[i].dot(sun).max(0.0);
        let absorbed = absorb_coeff * cos_incidence;
        let emitted = emit_coeff * ti * ti * ti * ti;

        // Lateral conduction: conductance-weighted sum of neighbor temperature differences.
        let neighbors = grid.neighbors(i);
        let weights = grid.neighbor_weights(i);
        let mut diffuse = 0.0f32;
        for (k, &j) in neighbors.iter().enumerate() {
            diffuse += weights[k] * (temp[j as usize] - ti);
        }

        let flux = absorbed - emitted + diffusivity * diffuse;
        *t_out = (ti + dt * flux * inv_c).max(temp_floor);
    });
}
