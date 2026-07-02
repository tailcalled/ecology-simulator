//! The per-cell grey-column energy-balance kernel — the hot loop, run in parallel with rayon.
//!
//! Each cell carries a surface skin temperature `T_s` and a column of `n` atmosphere layers
//! `T_0..T_{n-1}` (0 = lowest/warmest, n−1 = top/coldest). The layers are grey slabs: layer `l`
//! absorbs/emits a fraction `ε_l` of the longwave beam (transmitting `τ_l = 1−ε_l`) and radiates
//! `ε_l σ T_l⁴` both up and down. Longwave is resolved by a two-sweep beam bookkeeping through the
//! column — up from the surface, down from space (see [`longwave_column`]). Summed over the whole
//! column it collapses to `absorbed − OLR`, so the scheme conserves energy by construction. A
//! convective chain vents heat upward link-by-link (surface→0→1→…), each link firing only beyond a
//! per-link lapse threshold Γ, which builds the vertical structure (surface warmest, top coldest)
//! and a real greenhouse.
//!
//! Heat is transported laterally by the storm-track eddy conductance (on `T_s`) and the atmosphere's
//! freer mixing (on each layer), and **advected**: air layer `l` rides wind layer `l` in the
//! advective form `−u·∇T`, with the layer's global-mean tendency (`adv_corr[l]`) removed so
//! advection only redistributes heat. The surface skin is NOT advected (a sea/ground surface doesn't
//! blow downwind; horizontal heat reaches it via the advected air through convection). The per-cell
//! surface `C_s` (ocean ≫ land) damps the diurnal/seasonal swing.
//!
//! Reads come from the current fields; writes go to scratch buffers the caller swaps in, so neighbor
//! reads never observe a half-updated step. The kernel runs one parallel pass for the surface and
//! one per layer; each recomputes the (cheap, O(n)) column radiation, so per-cell cost is O(n²) —
//! negligible for the handful of layers used in practice, and it keeps each layer's storage
//! contiguous (trivial gradients and display slices).

use glam::Vec3;
use rayon::prelude::*;

use super::{Climate, MAX_LAYERS, STEFAN_BOLTZMANN};
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

/// Grey longwave for a column of `n` layers over the surface. Given the surface emission `b_s` and
/// each layer's emission `b[l] = ε_l σ T_l⁴`, does the up-sweep (from the surface) and the
/// down-sweep (from space) of the beam bookkeeping and writes each layer's net absorbed longwave
/// into `net`. Returns the downward flux reaching the surface.
///
/// With `τ_l = 1−ε_l`, the upward flux above interface `l` is `Uₗ₊₁ = τ_l·Uₗ + b_l` (starting from
/// the surface emission `U₀ = b_s`) and the downward flux below interface `l` is `Dₗ = τ_l·Dₗ₊₁ +
/// b_l` (starting from `D_n = 0` at space). Layer `l` absorbs `ε_l` of the beams entering its two
/// faces and emits `2·b_l`. Reduces to the hand-built two-layer exchange exactly.
fn longwave_column(b_s: f32, b: &[f32], emis: &[f32], tau: &[f32], n: usize, net: &mut [f32]) -> f32 {
    let mut up = [0.0f32; MAX_LAYERS + 1];
    up[0] = b_s; // upward flux leaving the surface
    for l in 0..n {
        up[l + 1] = tau[l] * up[l] + b[l];
    }
    let mut down = [0.0f32; MAX_LAYERS + 1];
    down[n] = 0.0; // no downwelling from space
    for l in (0..n).rev() {
        down[l] = tau[l] * down[l + 1] + b[l];
    }
    for l in 0..n {
        net[l] = emis[l] * (up[l] + down[l + 1]) - 2.0 * b[l];
    }
    down[0]
}

/// Advance the surface skin and the `n` atmosphere layers one explicit-Euler step.
///
/// `emis`/`heat_cap_atm` are the per-layer emissivity and areal heat capacity (length `n`);
/// `heat_cap_surf` is the per-cell surface value. `winds[l]`/`grads[l]` advect air layer `l`
/// (`grads[l]` is `∇T` of that layer), with `adv_corr[l]` the area-mean of `−u·∇T` subtracted for
/// conservation. Results go to `out_s`/`out_atm`; `out_conv` gets the surface→layer-0 convective
/// flux (a diagnostic).
#[allow(clippy::too_many_arguments)]
pub fn step(
    grid: &Grid,
    climate: &Climate,
    sun: Vec3,
    dt: f32,
    temp_floor: f32,
    gradient: f32,
    n: usize,
    temp_s: &[f32],
    temp_atm: &[Vec<f32>],
    emis: &[f32],
    heat_cap_atm: &[f32],
    heat_cap_surf: &[f32],
    winds: &[Vec<Vec3>],
    grads: &[Vec<Vec3>],
    adv_corr: &[f32],
    out_s: &mut [f32],
    out_atm: &mut [Vec<f32>],
    out_conv: &mut [f32],
) {
    let absorb_coeff = (1.0 - climate.albedo) * climate.solar_constant;
    let sigma = STEFAN_BOLTZMANN;
    let surf_emit = climate.emissivity * sigma;
    // Per-link lapse threshold: the total surface→top lapse split evenly over the `n` links, so the
    // overall vertical structure stays ~fixed as the column is refined into more layers.
    let gamma = climate.convection_lapse / n as f32;
    let atm_d = climate.atm_diffusivity;
    let advect_on = climate.atm_advection;

    let mut tau = [0.0f32; MAX_LAYERS];
    for l in 0..n {
        tau[l] = 1.0 - emis[l];
    }
    let tau = &tau[..n];

    // Per-link convective exchange, decaying with height: link 0 (surface→layer-0, the boundary
    // layer) keeps the full `convection_coeff`; higher links (the free troposphere) are weaker by
    // `convection_decay` per step, so the upper layers are only loosely tied to the surface and can
    // hold a warm anomaly long enough for the wind to carry it (heatwaves) — while the boundary
    // layer stays tightly coupled, preserving the diurnal cycle and the surface climate.
    let mut kappa = [0.0f32; MAX_LAYERS];
    let mut kk = climate.convection_coeff;
    for link in 0..n {
        kappa[link] = kk;
        kk *= climate.convection_decay;
    }

    // --- Surface pass: absorbed sunlight, net longwave onto the skin, venting into layer 0, and the
    // storm-track eddy diffusion. The skin is not advected. ---
    out_s
        .par_iter_mut()
        .zip(out_conv.par_iter_mut())
        .enumerate()
        .for_each(|(i, (o_s, o_c))| {
            let ts = temp_s[i];
            let ci = grid.centers[i];

            let mut b = [0.0f32; MAX_LAYERS];
            for l in 0..n {
                let tk = temp_atm[l][i];
                b[l] = emis[l] * sigma * tk * tk * tk * tk;
            }
            let b_s = surf_emit * ts * ts * ts * ts;
            let mut net = [0.0f32; MAX_LAYERS];
            let down_surf = longwave_column(b_s, &b, emis, tau, n, &mut net);

            let h0 = kappa[0] * (ts - temp_atm[0][i] - gamma).max(0.0);

            let cos_incidence = ci.dot(sun).max(0.0);
            let absorbed = absorb_coeff * cos_incidence;

            let ds_i = conductance(climate, gradient, ci.z);
            let neighbors = grid.neighbors(i);
            let weights = grid.neighbor_weights(i);
            let mut diffuse_s = 0.0f32;
            for (kk, &j) in neighbors.iter().enumerate() {
                let j = j as usize;
                let ds_j = conductance(climate, gradient, grid.centers[j].z);
                diffuse_s += 0.5 * (ds_i + ds_j) * weights[kk] * (temp_s[j] - ts);
            }

            let flux_s = absorbed + (down_surf - b_s) - h0 + diffuse_s;
            *o_s = (ts + dt * flux_s / heat_cap_surf[i]).max(temp_floor);
            *o_c = h0;
        });

    // --- One pass per atmosphere layer: net longwave, the two convective links bracketing it,
    // lateral mixing within the layer, and advection by that layer's wind. ---
    for l in 0..n {
        let inv_c = 1.0 / heat_cap_atm[l];
        let wind_l = &winds[l];
        let grad_l = &grads[l];
        let layer = &temp_atm[l];
        let corr = adv_corr[l];
        out_atm[l].par_iter_mut().enumerate().for_each(|(i, o)| {
            let ts = temp_s[i];

            let mut t = [0.0f32; MAX_LAYERS];
            let mut b = [0.0f32; MAX_LAYERS];
            for k in 0..n {
                let tk = temp_atm[k][i];
                t[k] = tk;
                b[k] = emis[k] * sigma * tk * tk * tk * tk;
            }
            let b_s = surf_emit * ts * ts * ts * ts;
            let mut net = [0.0f32; MAX_LAYERS];
            let _ = longwave_column(b_s, &b, emis, tau, n, &mut net);

            // Convective links: heat received through the link below (index `l`), lost through the
            // link above (index `l+1`), each with its own height-decayed strength.
            let h_in = if l == 0 {
                kappa[0] * (ts - t[0] - gamma).max(0.0)
            } else {
                kappa[l] * (t[l - 1] - t[l] - gamma).max(0.0)
            };
            let h_out = if l + 1 < n {
                kappa[l + 1] * (t[l] - t[l + 1] - gamma).max(0.0)
            } else {
                0.0
            };

            let tl = t[l];
            let neighbors = grid.neighbors(i);
            let weights = grid.neighbor_weights(i);
            let mut diffuse = 0.0f32;
            for (kk, &j) in neighbors.iter().enumerate() {
                diffuse += atm_d * weights[kk] * (layer[j as usize] - tl);
            }

            let advect = if advect_on { -wind_l[i].dot(grad_l[i]) - corr } else { 0.0 };
            let flux = net[l] + h_in - h_out + diffuse;
            *o = (tl + dt * (flux * inv_c + advect)).max(temp_floor);
        });
    }
}
