//! The per-cell three-body energy-balance kernel — the hot loop, run in parallel with rayon.
//!
//! Three temperatures evolve per cell: the surface `T_s`, the **lower** atmosphere `T_l`, and the
//! **upper** atmosphere `T_u`, exchanging energy by grey-body longwave radiation and a convective
//! chain. The atmosphere is two stacked grey slabs (emissivities ε_L, ε_U), each absorbing a
//! fraction of the longwave beam passing through it and re-emitting σεT⁴ both up and down. With the
//! beam bookkeeping done (τ = 1−ε per layer):
//!
//! ```text
//!   surface:  C_s dT_s/dt = (1−α)S·max(0,n·ŝ) − σT_s⁴ + (τ_L·σε_UT_u⁴ + σε_LT_l⁴) − H_s + dif_s
//!   lower:    C_L dT_l/dt = ε_L(σT_s⁴ + σε_UT_u⁴) − 2σε_LT_l⁴ + H_s − H_l + dif_l − u·∇T_l
//!   upper:    C_U dT_u/dt = ε_U(τ_L·σT_s⁴ + σε_LT_l⁴) − 2σε_UT_u⁴ + H_l + dif_u − u_hi·∇T_u
//!   convection chain:  H_s = κ·max(T_s−T_l−Γ, 0) ,  H_l = κ·max(T_l−T_u−Γ, 0)
//! ```
//!
//! Summing the three budgets, the radiative terms collapse to `absorbed − OLR` with
//! `OLR = τ_U(τ_L σT_s⁴ + σε_LT_l⁴) + σε_UT_u⁴` and the convective fluxes cancel — the scheme
//! conserves energy by construction, and the cold upper layer plus the per-link lapse threshold Γ
//! give an emergent vertical structure (surface warmest, upper coldest) and a real **greenhouse**.
//!
//! Heat is transported laterally by the storm-track eddy conductance (`dif_s`, on `T_s`) and the
//! atmosphere's freer mixing (`dif_l`/`dif_u`), and **advected**: the *air* rides the winds — `T_l`
//! the surface wind, `T_u` the upper wind. The surface skin is NOT advected (a sea/ground surface
//! doesn't blow downwind; horizontal heat reaches it via the advected air through convection). The
//! per-cell `C_s` (ocean ≫ land) is what damps the diurnal/seasonal swing. Advection uses the
//! (locally stable) advective form `−u·∇T`; the upper layer subtracts the global-mean tendency
//! (`adv_u_correction`) so its fast, divergent wind doesn't leak a spurious global drift (the stand-in
//! for vertical mass compensation). The two layers' opposed overturning winds carrying *different*
//! temperatures is what lets the circulation transport heat poleward — impossible with a single slab.
//!
//! Reads come from the `Temps` snapshots; writes go to the `Outs` scratch buffers. The caller swaps
//! afterward, so neighbor reads never observe a half-updated step.

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

/// The three current temperature fields (read-only snapshots) the kernel evolves.
pub struct Temps<'a> {
    /// Surface temperature `T_s` (K).
    pub s: &'a [f32],
    /// Lower atmosphere temperature `T_l` (K).
    pub l: &'a [f32],
    /// Upper atmosphere temperature `T_u` (K).
    pub u: &'a [f32],
}

/// The output scratch buffers the kernel writes (caller swaps them in afterward).
pub struct Outs<'a> {
    /// Next surface temperature.
    pub s: &'a mut [f32],
    /// Next lower-atmosphere temperature.
    pub l: &'a mut [f32],
    /// Next upper-atmosphere temperature.
    pub u: &'a mut [f32],
    /// Surface→lower convective flux H_s (W·m⁻²) for visualization.
    pub conv: &'a mut [f32],
}

/// Advance the surface and two atmosphere temperatures one explicit-Euler step. `wind`/`grad_s`
/// advect the surface and `wind`/`grad_l` the lower layer; `wind_hi`/`grad_u` advect the upper
/// layer (with `adv_u_correction`, the area-mean of `−u_hi·∇T_u`, subtracted to keep it
/// energy-conserving). `gradient` is the shared equator-to-pole factor. Results + the surface
/// convective flux go to `out`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    grid: &Grid,
    climate: &Climate,
    sun: Vec3,
    dt: f32,
    temp_floor: f32,
    gradient: f32,
    temps: Temps,
    heat_cap: &[f32],
    wind: &[Vec3],
    grad_l: &[Vec3],
    wind_hi: &[Vec3],
    grad_u: &[Vec3],
    adv_u_correction: f32,
    out: Outs,
) {
    let absorb_coeff = (1.0 - climate.albedo) * climate.solar_constant;
    let sigma = STEFAN_BOLTZMANN;
    let surf_emit = climate.emissivity * sigma;
    let eps_l = climate.atm_emissivity_lower;
    let eps_u = climate.atm_emissivity_upper;
    let tau_l = 1.0 - eps_l; // lower-layer longwave transmissivity
    let inv_cl = 1.0 / climate.atm_heat_capacity_lower;
    let inv_cu = 1.0 / climate.atm_heat_capacity_upper;
    let kappa = climate.convection_coeff;
    let gamma = climate.convection_threshold;
    let atm_d = climate.atm_diffusivity;

    let (Temps { s: temp_s, l: temp_l, u: temp_u }, Outs { s: out_s, l: out_l, u: out_u, conv: out_conv }) =
        (temps, out);

    out_s
        .par_iter_mut()
        .zip(out_l.par_iter_mut())
        .zip(out_u.par_iter_mut())
        .zip(out_conv.par_iter_mut())
        .enumerate()
        .for_each(|(i, (((ts_out, tl_out), tu_out), conv_out))| {
            let ts = temp_s[i];
            let tl = temp_l[i];
            let tu = temp_u[i];
            let ci = grid.centers[i];

            let cos_incidence = ci.dot(sun).max(0.0);
            let absorbed = absorb_coeff * cos_incidence;

            // Grey-body emissions (per direction for the layers).
            let b_s = surf_emit * ts * ts * ts * ts;
            let b_l = eps_l * sigma * tl * tl * tl * tl;
            let b_u = eps_u * sigma * tu * tu * tu * tu;

            // Net longwave for each body, from the up/down beam bookkeeping (see module docs).
            let lw_s = (tau_l * b_u + b_l) - b_s;
            let lw_l = eps_l * (b_s + b_u) - 2.0 * b_l;
            let lw_u = eps_u * (tau_l * b_s + b_l) - 2.0 * b_u;

            // Convection chain: surface vents into the lower layer, lower vents into the upper, each
            // only beyond the per-link lapse threshold Γ (keeps each layer cooler than the one below).
            let h_s = kappa * (ts - tl - gamma).max(0.0);
            let h_l = kappa * (tl - tu - gamma).max(0.0);

            // Lateral transport: surface storm-track eddy conductance; both atmosphere layers use
            // the freer atmospheric mixing. Edge conductances average the two cells' values so each
            // operator stays conservative.
            let ds_i = conductance(climate, gradient, ci.z);
            let neighbors = grid.neighbors(i);
            let weights = grid.neighbor_weights(i);
            let mut diffuse_s = 0.0f32;
            let mut diffuse_l = 0.0f32;
            let mut diffuse_u = 0.0f32;
            for (k, &j) in neighbors.iter().enumerate() {
                let j = j as usize;
                let ds_j = conductance(climate, gradient, grid.centers[j].z);
                diffuse_s += 0.5 * (ds_i + ds_j) * weights[k] * (temp_s[j] - ts);
                diffuse_l += atm_d * weights[k] * (temp_l[j] - tl);
                diffuse_u += atm_d * weights[k] * (temp_u[j] - tu);
            }

            // Advection (advective form; the conduction terms keep the central gradient stable).
            // Only the AIR is advected — the lower layer by the surface wind, the upper by the upper
            // wind (with its divergent global drift removed). The surface skin is NOT advected: a
            // sea/ground surface doesn't blow downwind with the air (its heat moves with ocean
            // currents, ~100× slower), and advecting a high-`C_s` ocean cell by the full atmospheric
            // wind is both unphysical and numerically unstable (the advective tendency doesn't shrink
            // with `C_s` but its stabilizing diffusion does). Horizontal heat reaches the surface via
            // the advected air (convection) + the storm-track surface diffusion.
            let advect_l = -wind[i].dot(grad_l[i]);
            let advect_u = if climate.upper_advection {
                -wind_hi[i].dot(grad_u[i]) - adv_u_correction
            } else {
                0.0
            };

            let flux_s = absorbed + lw_s - h_s + diffuse_s;
            let flux_l = lw_l + h_s - h_l + diffuse_l;
            let flux_u = lw_u + h_l + diffuse_u;

            let inv_cs = 1.0 / heat_cap[i]; // ocean cells (high C_s) barely swing; land swings fast
            *ts_out = (ts + dt * flux_s * inv_cs).max(temp_floor);
            *tl_out = (tl + dt * (flux_l * inv_cl + advect_l)).max(temp_floor);
            *tu_out = (tu + dt * (flux_u * inv_cu + advect_u)).max(temp_floor);
            *conv_out = h_s;
        });
}
