//! The per-cell two-layer energy-balance kernel — the hot loop, run in parallel with rayon.
//!
//! Two coupled temperatures evolve: the surface `T_s` and the atmosphere `T_a`, exchanging energy
//! by grey-body radiation and convection. Per cell, over one step `dt` (σ = Stefan–Boltzmann,
//! ε_a = atmospheric IR emissivity):
//!
//!   surface:     C_s·dT_s/dt = (1−α)S·max(0,n·ŝ) − σT_s⁴ + ε_a·σT_a⁴ − H + diffuse_s ;  −u·∇T_s
//!   atmosphere:  C_a·dT_a/dt = ε_a·σT_s⁴ + H − 2ε_a·σT_a⁴ + diffuse_a
//!   convection:  H = κ·(T_s − T_a)                         [sensible-heat flux, the vertical venting]
//!
//! This is a single grey atmospheric slab: it absorbs a fraction ε_a of the surface's thermal
//! radiation and re-emits σε_a·T_a⁴ both up (to space) and down (back-radiation to the surface),
//! which is the **greenhouse** — it lifts the surface above its bare-rock emission temperature, so
//! the old emissivity fudge is gone. The convective flux `H` carries heat up where the surface is
//! hot (the **venting**, conservative: the surface loses exactly what the air gains), and the
//! atmosphere's large heat capacity `C_a` buffers the diurnal swing.
//!
//! `diffuse_s` carries the storm-track eddy transport that warms the poles (as before); `diffuse_a`
//! is the atmosphere's freer lateral mixing. Surface heat is advected by the resolved surface wind
//! in **advective** form `−u·∇T_s` (the flux form's `−T∇·u` would pile heat up unboundedly in
//! convergence zones — that upward escape is now handled physically by convection into the
//! atmosphere instead).
//!
//! Reads come from the `*_in` snapshots; writes go to the `*_out` scratch buffers. The caller
//! swaps afterward, so neighbor reads never observe a half-updated step.

use glam::Vec3;
use rayon::prelude::*;

use super::{Climate, STEFAN_BOLTZMANN};
use crate::grid::Grid;

/// Latitude (radians) of the storm-track peak — the mid-latitude baroclinic zone where eddy heat
/// transport is strongest.
const STORM_TRACK_LAT: f32 = 0.873; // ≈ 50°
/// Gaussian half-width (radians) of the storm-track band around [`STORM_TRACK_LAT`]. Broad enough
/// that the eddy transport still reaches into the polar cap (the flux must converge poleward of the
/// storm-track maximum to actually warm the poles).
const STORM_TRACK_WIDTH: f32 = 0.52; // ≈ 30°

/// Mid-to-high-latitude eddy-transport profile: a Gaussian bump peaking at 1.0 in the storm track,
/// fading toward the equator (where the mean Hadley cell, not eddies, dominates) and broad on the
/// poleward side so heat is carried into the caps. `sin_lat` is `sin φ` of the cell.
fn storm_track(sin_lat: f32) -> f32 {
    let lat = sin_lat.clamp(-1.0, 1.0).asin();
    let d = (lat.abs() - STORM_TRACK_LAT) / STORM_TRACK_WIDTH;
    (-d * d).exp()
}

/// Per-cell lateral conductance: base conduction plus the storm-track eddy term, scaled by the
/// climate `gradient` factor. `sin_lat` is the cell's `sin φ`.
fn conductance(climate: &Climate, gradient: f32, sin_lat: f32) -> f32 {
    climate.diffusivity + climate.eddy_diffusivity * gradient * storm_track(sin_lat)
}

/// Advance the surface and atmosphere temperatures one explicit-Euler step. `temp_s`/`temp_a` are
/// the current fields; `wind` (m·s⁻¹) advects the surface and `grad_t` (K·m⁻¹) is its gradient;
/// `gradient` is the shared equator-to-pole factor. Results go to `out_s`/`out_a`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    grid: &Grid,
    climate: &Climate,
    sun: Vec3,
    dt: f32,
    temp_floor: f32,
    gradient: f32,
    temp_s: &[f32],
    temp_a: &[f32],
    wind: &[Vec3],
    grad_t: &[Vec3],
    out_s: &mut [f32],
    out_a: &mut [f32],
) {
    let absorb_coeff = (1.0 - climate.albedo) * climate.solar_constant;
    let sigma = STEFAN_BOLTZMANN;
    let surf_emit = climate.emissivity * sigma;
    let eps_a = climate.atm_emissivity;
    let atm_emit = eps_a * sigma;
    let inv_cs = 1.0 / climate.heat_capacity;
    let inv_ca = 1.0 / climate.atm_heat_capacity;
    let kappa = climate.convection_coeff;
    let gamma = climate.convection_threshold;
    let atm_d = climate.atm_diffusivity;

    out_s
        .par_iter_mut()
        .zip(out_a.par_iter_mut())
        .enumerate()
        .for_each(|(i, (ts_out, ta_out))| {
            let ts = temp_s[i];
            let ta = temp_a[i];
            let ci = grid.centers[i];

            let cos_incidence = ci.dot(sun).max(0.0);
            let absorbed = absorb_coeff * cos_incidence;
            let ts4 = ts * ts * ts * ts;
            let ta4 = ta * ta * ta * ta;
            let surf_up = surf_emit * ts4; // surface thermal emission
            let atm_ir = atm_emit * ta4; // atmosphere emission (each of up and down)

            // Convective sensible-heat flux, upward, only once the surface is warmer than the air by
            // more than the lapse threshold — so the air stays ~Γ cooler than the surface and the
            // greenhouse survives.
            let h = kappa * (ts - ta - gamma).max(0.0);

            // Lateral transport: surface storm-track eddy conductance; atmosphere's freer mixing.
            // Edge conductances average the two cells' values so each operator stays conservative.
            let ds_i = conductance(climate, gradient, ci.z);
            let neighbors = grid.neighbors(i);
            let weights = grid.neighbor_weights(i);
            let mut diffuse_s = 0.0f32;
            let mut diffuse_a = 0.0f32;
            for (k, &j) in neighbors.iter().enumerate() {
                let j = j as usize;
                let ds_j = conductance(climate, gradient, grid.centers[j].z);
                diffuse_s += 0.5 * (ds_i + ds_j) * weights[k] * (temp_s[j] - ts);
                diffuse_a += atm_d * weights[k] * (temp_a[j] - ta);
            }

            // Surface advection by the resolved wind, −(u·∇T_s); the strong conduction term keeps
            // this central estimate stable.
            let advect_s = -wind[i].dot(grad_t[i]);

            let flux_s = absorbed - surf_up + atm_ir - h + diffuse_s;
            let flux_a = eps_a * surf_up + h - 2.0 * atm_ir + diffuse_a;

            *ts_out = (ts + dt * (flux_s * inv_cs + advect_s)).max(temp_floor);
            *ta_out = (ta + dt * (flux_a * inv_ca)).max(temp_floor);
        });
}
