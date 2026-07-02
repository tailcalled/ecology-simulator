//! Surface wind: a diagnostic vector field recomputed each step from the current temperature
//! field and the planet's geometry. There is no momentum state — the wind is in instantaneous
//! balance with the climate, which is the right level of fidelity for an ecology model and keeps
//! the update cheap and unconditionally stable.
//!
//! The field is the sum of two parts:
//!
//!   1. **Tri-cell zonal-mean belts** (`belt_wind`). A single vertically-integrated layer cannot
//!      *produce* the Hadley/Ferrel/Polar overturning (that needs a vertical dimension), so the
//!      robust, well-understood belt structure is imposed analytically: easterly trades (0–30°),
//!      mid-latitude westerlies (30–60°), polar easterlies (60–90°), with a matching meridional
//!      surface branch (equatorward in the Hadley/Polar cells, poleward in the Ferrel cell) so the
//!      trades come out north*easterly* as on Earth. Crucially the belt *strength* is scaled by
//!      the **simulated** equator-to-pole temperature contrast (`gradient_factor`), so the winds
//!      strengthen and weaken with the climate and the seasons rather than being painted on.
//!
//!   2. **Geostrophic perturbation** (`geostrophic_wind`). The departures from the zonal mean —
//!      monsoon flow, continental highs and lows — come from the actual temperature field: treat
//!      cold air as high pressure, take the surface temperature gradient, and balance the
//!      pressure-gradient force against the Coriolis force (`f = 2Ω sinφ`, capped near the equator
//!      where geostrophy breaks down), with a friction term turning the flow partly across the
//!      isobars toward low pressure.
//!
//! Everything is a tangent vector to the unit sphere (perpendicular to the cell normal), in m·s⁻¹,
//! mirroring [`Terrain::velocity`](super::terrain::Terrain::velocity).

use glam::Vec3;
use rayon::prelude::*;

use super::Climate;
use crate::grid::Grid;

/// Half-width (radians of latitude) of the equatorial belt averaged for the "equator" end of the
/// climate gradient, and likewise the polar cap for the "pole" end.
const EQUATOR_BAND: f32 = 15.0;
const POLE_BAND: f32 = 75.0;

/// Reference equator-to-pole temperature contrast (K) at which the belts blow at their nominal
/// speed. Roughly Earth's present-day surface value (~300 K equator, ~250 K pole). The belts
/// scale linearly with the simulated contrast relative to this.
const REFERENCE_GRADIENT: f32 = 50.0;

/// Hard cap on wind speed (m·s⁻¹) for the prognostic integration — a physical backstop (Earth's
/// fastest jets ~110 m·s⁻¹) that also guarantees the temperature-advection CFL can never be
/// violated, so a transient gradient spike can't blow the sim up.
const MAX_WIND: f32 = 120.0;

/// Orthonormal surface frame at a cell normal `n`: (east, north). `east × north = n` (a
/// right-handed up/north/east triad). Degenerates at the poles, where the returned directions are
/// arbitrary but still orthonormal and tangent.
pub fn east_north(n: Vec3) -> (Vec3, Vec3) {
    // Geographic north is the component of +Z tangent to the surface.
    let north = (Vec3::Z - n * n.z).normalize_or_zero();
    let north = if north.length_squared() > 0.5 {
        north
    } else {
        // At a pole, pick any tangent direction.
        Vec3::X.cross(n).normalize_or_zero()
    };
    let east = north.cross(n).normalize_or_zero();
    (east, north)
}

/// Area-weighted equator-to-pole temperature contrast (K) of the current field, divided by the
/// reference contrast. 1.0 ≈ today's Earth; larger means a steeper gradient (stronger circulation),
/// 0 a flat planet (no thermally-driven wind). Clamped to keep the belts bounded.
pub fn gradient_factor(grid: &Grid, temp: &[f32]) -> f32 {
    let (mut eq_t, mut eq_a, mut pole_t, mut pole_a) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..grid.n {
        let lat = grid.lonlat_deg[i].y.abs();
        let a = grid.areas[i] as f64;
        if lat < EQUATOR_BAND {
            eq_t += temp[i] as f64 * a;
            eq_a += a;
        } else if lat > POLE_BAND {
            pole_t += temp[i] as f64 * a;
            pole_a += a;
        }
    }
    if eq_a <= 0.0 || pole_a <= 0.0 {
        return 1.0;
    }
    let contrast = (eq_t / eq_a - pole_t / pole_a) as f32;
    (contrast / REFERENCE_GRADIENT).clamp(0.0, 2.0)
}

/// Least-squares tangent-plane gradient of a per-cell scalar `field` at every cell, in units of
/// `field` per metre (using `radius` to turn the unit-sphere arc lengths into physical distance).
///
/// For each cell we fit the planar model `field(x) ≈ field(i) + g·x` over the neighbor
/// displacements `x` (projected into the cell's tangent plane), solving the 2×2 normal equations.
/// The result `g` is returned as a tangent vector (its direction is up-gradient, its length the
/// slope). Cells with too few neighbors to fit get a zero gradient.
pub fn gradients(grid: &Grid, field: &[f32], radius: f32) -> Vec<Vec3> {
    (0..grid.n)
        .into_par_iter()
        .map(|i| {
            let n = grid.centers[i];
            let (east, north) = east_north(n);
            let ti = field[i];

            // Accumulate the normal-equation matrix [a b; b c] and rhs (rx, ry) in the (east,
            // north) basis.
            let (mut a, mut b, mut c, mut rx, mut ry) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
            let dists = grid.neighbor_dists(i);
            for (kk, &j) in grid.neighbors(i).iter().enumerate() {
                let pj = grid.centers[j as usize];
                // Tangent-plane displacement to neighbor j, scaled to the great-circle distance.
                let tang = pj - n * pj.dot(n);
                let dir = tang.normalize_or_zero();
                if dir.length_squared() < 0.5 {
                    continue;
                }
                let dist = dists[kk] * radius;
                let d = dir * dist;
                let (dx, dy) = (d.dot(east), d.dot(north));
                let dt = field[j as usize] - ti;
                a += dx * dx;
                b += dx * dy;
                c += dy * dy;
                rx += dx * dt;
                ry += dy * dt;
            }

            let det = a * c - b * b;
            if det.abs() < 1e-12 {
                return Vec3::ZERO;
            }
            let gx = (c * rx - b * ry) / det;
            let gy = (a * ry - b * rx) / det;
            east * gx + north * gy
        })
        .collect()
}

/// Latitude of the subtropical jet maximum (radians ≈ 30°): the poleward edge of the Hadley cell,
/// where the upper-level westerlies peak.
const JET_LAT: f32 = std::f32::consts::FRAC_PI_6;
/// Gaussian half-width (radians ≈ 20°) of the subtropical jet in latitude.
const JET_WIDTH: f32 = 0.35;

/// Normalized height of wind layer `layer` in an `n_layers`-deep column: 0 at the surface layer, 1
/// at the top. A single-layer column sits at the surface.
pub fn layer_height(layer: usize, n_layers: usize) -> f32 {
    if n_layers <= 1 {
        0.0
    } else {
        layer as f32 / (n_layers - 1) as f32
    }
}

/// The zonal-mean belt target for a wind layer at normalized height `h` (0 = surface, 1 = top),
/// generalizing the two hand-built belts to a column of any depth (scaled by the climate `gradient`
/// factor). A single vertically-integrated layer cannot *produce* the Hadley/Ferrel/Polar
/// overturning, so its zonal-mean structure is imposed here as a vertical profile:
///
///   * **Zonal** blends the surface **tri-cell** profile at `h=0` (`−sin(6|φ|)`: easterly trades
///     0–30°, mid-latitude westerlies 30–60°, polar easterlies 60–90°) into the upper **subtropical
///     jet** at `h=1` (a single westerly Gaussian peaking at [`JET_LAT`] ≈ 30°).
///   * **Meridional** is the overturning branch, ramped linearly from equatorward at `h=0` to
///     poleward at `h=1` via the `(2h−1)` factor. Being antisymmetric about mid-column its vertical
///     integral is ~zero, so the imposed overturning conserves mass and adjacent-height branches
///     automatically oppose — closing the cells at every layer for any depth. (This is the vertical
///     derivative of a linear overturning streamfunction.)
///
/// Reduces exactly to the old two-layer belts at `h ∈ {0, 1}`.
fn belt_for_layer(h: f32, n: Vec3, east: Vec3, north: Vec3, climate: &Climate, gradient: f32) -> Vec3 {
    let lat = n.z.clamp(-1.0, 1.0).asin();
    let shape = (6.0 * lat.abs()).sin();
    // Zonal: surface tri-cell (h=0) → subtropical jet (h=1).
    let zonal_surface = -climate.trade_wind_speed * shape;
    let d = (lat.abs() - JET_LAT) / JET_WIDTH;
    let zonal_jet = climate.jet_wind_speed * (-d * d).exp();
    let zonal = zonal_surface * (1.0 - h) + zonal_jet * h;
    // Meridional overturning: equatorward (−) at the surface, poleward (+) aloft, linear in height.
    let poleward = climate.meridional_wind_speed * shape * lat.signum() * (2.0 * h - 1.0);
    (east * zonal + north * poleward) * gradient
}

/// The surface tri-cell belt (the `h=0` end of [`belt_for_layer`]). Used by the belt-shape tests.
#[cfg(test)]
fn belt_wind(n: Vec3, east: Vec3, north: Vec3, climate: &Climate, gradient: f32) -> Vec3 {
    belt_for_layer(0.0, n, east, north, climate, gradient)
}

/// The upper-troposphere belt — subtropical jet + overturning return branch (the `h=1` end of
/// [`belt_for_layer`]). Used by the belt-shape tests.
#[cfg(test)]
fn belt_wind_upper(n: Vec3, east: Vec3, north: Vec3, climate: &Climate, gradient: f32) -> Vec3 {
    belt_for_layer(1.0, n, east, north, climate, gradient)
}

/// The wind perturbation from the local temperature gradient, via a steady Ekman balance of three
/// accelerations: the pressure-gradient force, the Coriolis force, and linear surface friction.
///
/// Warm air ⇒ low pressure, so the pressure-gradient acceleration points *up* the temperature
/// gradient (toward the warm low): `a = k·∇T`, where `k = geostrophic_coeff` carries the
/// pressure-per-kelvin / density conversion so `a` is a real m·s⁻² (this scale was the missing
/// factor that left the perturbation invisibly small before). Balancing `a = f(n×v) + r·v` and
/// solving for `v` gives
///
/// ```text
///     v = (r·a − f·(n×a)) / (r² + f²)
/// ```
///
/// which reduces to the geostrophic wind (along the isobars) where friction `r` is weak, turns the
/// flow across the isobars toward the low as `r` grows, and — because `r² + f²` never vanishes —
/// stays finite straight across the equator, so no Coriolis cap is needed.
fn geostrophic_wind(n: Vec3, grad_t: Vec3, climate: &Climate) -> Vec3 {
    let lat = n.z.clamp(-1.0, 1.0).asin();
    // Coriolis parameter f = 2Ω sinφ (signed), with Ω from the rotation period.
    let omega = std::f32::consts::TAU / climate.day_seconds;
    let f = 2.0 * omega * lat.sin();
    let r = climate.wind_friction;
    let denom = r * r + f * f;
    if denom < 1e-20 {
        return Vec3::ZERO;
    }
    let accel = grad_t * climate.geostrophic_coeff; // toward warm = toward low pressure
    (accel * r - n.cross(accel) * f) / denom
}

/// The instantaneous balance wind (belts + geostrophic), into `out`. Used only to *seed* the
/// prognostic field with a sensible starting state (see [`step`]); the live wind is then evolved by
/// integrating momentum. `grad_t` is the temperature gradient (from [`gradients`]) and `gradient`
/// the climate factor (from [`gradient_factor`]). Each entry is a tangent vector in m·s⁻¹. `layer`
/// `layer`/`n_layers` place this wind at its normalized height (0 = surface … 1 = top), which
/// selects its belt target; `grad_t` is the gradient driving its pressure force.
pub fn compute(
    grid: &Grid,
    climate: &Climate,
    gradient: f32,
    grad_t: &[Vec3],
    out: &mut [Vec3],
    layer: usize,
    n_layers: usize,
) {
    let h = layer_height(layer, n_layers);
    out.par_iter_mut().enumerate().for_each(|(i, w)| {
        let n = grid.centers[i];
        let (east, north) = east_north(n);
        let belt = belt_for_layer(h, n, east, north, climate, gradient);
        let geo = geostrophic_wind(n, grad_t[i], climate);
        let v = belt + geo;
        // Re-project to the tangent plane to kill any tiny normal component from summation.
        *w = v - n * v.dot(n);
    });
}

/// Number of latitude bands used to separate the zonal mean (the tri-cell climatology) from the
/// eddies. ~3° each over the 16k-cell grid: fine enough to resolve the belts, coarse enough that
/// each band still averages over many cells.
const ZONAL_BANDS: usize = 60;

/// Per-band zonal means needed to split the wind and the pressure forcing into "mean" (belts) and
/// "eddy" (weather) parts. Each is the area-weighted mean over a latitude band of the named
/// quantity's eastward and northward components.
struct ZonalMeans {
    wind_e: [f32; ZONAL_BANDS],
    wind_n: [f32; ZONAL_BANDS],
    press_e: [f32; ZONAL_BANDS],
    press_n: [f32; ZONAL_BANDS],
}

/// Latitude-band index (0..ZONAL_BANDS) of a cell normal.
fn band_of(n: Vec3) -> usize {
    let lat = n.z.clamp(-1.0, 1.0).asin();
    let t = (lat / std::f32::consts::PI + 0.5).clamp(0.0, 1.0); // 0 at S pole, 1 at N pole
    ((t * ZONAL_BANDS as f32) as usize).min(ZONAL_BANDS - 1)
}

/// Compute the per-band zonal means of the wind and of the pressure forcing `k·∇T`.
fn zonal_means(grid: &Grid, u: &[Vec3], grad_t: &[Vec3], k: f32) -> ZonalMeans {
    let mut wind_e = [0.0f32; ZONAL_BANDS];
    let mut wind_n = [0.0f32; ZONAL_BANDS];
    let mut press_e = [0.0f32; ZONAL_BANDS];
    let mut press_n = [0.0f32; ZONAL_BANDS];
    let mut area = [0.0f32; ZONAL_BANDS];
    for i in 0..grid.n {
        let n = grid.centers[i];
        let (east, north) = east_north(n);
        let b = band_of(n);
        let a = grid.areas[i];
        wind_e[b] += u[i].dot(east) * a;
        wind_n[b] += u[i].dot(north) * a;
        let press = grad_t[i] * k;
        press_e[b] += press.dot(east) * a;
        press_n[b] += press.dot(north) * a;
        area[b] += a;
    }
    for b in 0..ZONAL_BANDS {
        if area[b] > 0.0 {
            let inv = 1.0 / area[b];
            wind_e[b] *= inv;
            wind_n[b] *= inv;
            press_e[b] *= inv;
            press_n[b] *= inv;
        }
    }
    ZonalMeans { wind_e, wind_n, press_e, press_n }
}

/// Advance the **prognostic** surface wind one explicit step. The wind is split into its zonal mean
/// (the tri-cell climatology) and the eddies (weather), which are forced differently so that the
/// cells survive while the eddies are free to turbulence:
///
/// ```text
///   ∂u/∂t = −(u·∇)u − f(n×u) + k·∇T' − μ·u − R·(ū − u_belt) + ν·∇²u
/// ```
///
/// where `ū` is the zonal-mean wind and `∇T'` is the pressure forcing with its zonal mean removed.
/// Removing the mean from the pressure force is essential: the raw surface temperature gradient,
/// balanced geostrophically, gives *easterlies at every latitude* (the pole-cold high-pressure
/// response), which would wipe out the westerly belt — so the zonal-mean circulation is set by the
/// belt relaxation `R·(ū − u_belt)` instead, and the pressure force drives only the eddies. `μ` is a
/// light Rayleigh drag (eddy dissipation + equatorial bound), `ν` the eddy viscosity, and the
/// nonlinear self-advection is the eddy energy cascade.
#[allow(clippy::too_many_arguments)]
pub fn step(
    grid: &Grid,
    climate: &Climate,
    gradient: f32,
    grad_t: &[Vec3],
    dt: f32,
    u: &[Vec3],
    out: &mut [Vec3],
    layer: usize,
    n_layers: usize,
    couple_to: Option<&[Vec3]>,
) {
    let radius = climate.planet_radius;
    let coupling = climate.vertical_wind_coupling;
    let omega = std::f32::consts::TAU / climate.day_seconds;
    let zonal_relax = climate.wind_relax;
    let h = layer_height(layer, n_layers);
    // The surface layer keeps its full Rayleigh friction (eddy dissipation + equatorial bound); the
    // free-tropospheric layers aloft feel little surface drag, so their jet can run fast.
    let friction = if layer == 0 {
        climate.wind_friction
    } else {
        climate.wind_friction_upper
    };
    let visc = climate.wind_viscosity;
    let k = climate.geostrophic_coeff;

    let zm = zonal_means(grid, u, grad_t, k);

    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let n = grid.centers[i];
        let ui = u[i];
        let (east, north) = east_north(n);
        let belt = belt_for_layer(h, n, east, north, climate, gradient);
        let b = band_of(n);

        // Nonlinear self-advection in conservative flux form −∇·(u u), with antisymmetric, upwinded
        // edge fluxes (same scheme as the temperature kernel). Upwinding the carried momentum gives
        // the dissipation that keeps explicit time-stepping stable — a central estimate blows up.
        // The viscous Laplacian is accumulated in the same neighbor loop.
        let neighbors = grid.neighbors(i);
        let weights = grid.neighbor_weights(i);
        let dists = grid.neighbor_dists(i);
        let mut adv_flux = Vec3::ZERO;
        let mut lap = Vec3::ZERO;
        for (kk, &j) in neighbors.iter().enumerate() {
            let j = j as usize;
            let cj = grid.centers[j];
            lap += weights[kk] * (u[j] - ui);

            let dir = (cj - n * cj.dot(n)).normalize_or_zero();
            if dir.length_squared() > 0.5 {
                // Shared Voronoi edge length (radians) = conductance · center-to-center distance.
                let edge_arc = weights[kk] * dists[kk];
                let vel = 0.5 * (ui + u[j]).dot(dir);
                let u_up = if vel >= 0.0 { ui } else { u[j] };
                adv_flux += edge_arc * vel * u_up;
            }
        }
        let adv = adv_flux / (grid.areas[i] * radius); // = ∇·(u u); subtracted below

        // Coriolis acceleration −f(n×u).
        let f = 2.0 * omega * n.z;
        let coriolis = n.cross(ui) * f;

        // Pressure force with its zonal mean removed → drives only the eddies, not the (spurious)
        // easterly zonal-mean response.
        let press_full = grad_t[i] * k;
        let press_mean = zm.press_e[b] * east + zm.press_n[b] * north;
        let pressure = press_full - press_mean;

        // Zonal-mean relaxation: nudge the band's mean wind toward the belt target. Applied uniformly
        // across the band, it shifts the mean without touching the eddies (the departures from it).
        let belt_e = belt.dot(east);
        let belt_n = belt.dot(north);
        let zonal_correction =
            ((belt_e - zm.wind_e[b]) * east + (belt_n - zm.wind_n[b]) * north) * zonal_relax;

        // Light Rayleigh drag on the full wind: dissipates eddies and bounds the wind at the equator.
        let drag = -ui * friction;

        // Vertical momentum coupling: relax toward the other layer's wind (the upper layer toward the
        // turbulent surface wind), representing vertical mixing / the barotropic component. Injects
        // the surface's variability aloft so the jet is lively rather than steady.
        let vertical = match couple_to {
            Some(other) => (other[i] - ui) * coupling,
            None => Vec3::ZERO,
        };

        let accel = -adv - coriolis + pressure + zonal_correction + drag + lap * visc + vertical;
        let mut un = ui + accel * dt;
        // Keep the wind tangent to the sphere.
        un -= n * un.dot(n);
        // Physical speed cap (also guarantees CFL stability of the downstream heat advection).
        let speed = un.length();
        if speed > MAX_WIND {
            un *= MAX_WIND / speed;
        }
        *o = un;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f32 = 6.371e6;

    /// East/north frame is orthonormal, tangent, and right-handed (east × north = up) away from
    /// the poles.
    #[test]
    fn frame_is_orthonormal_and_tangent() {
        for n in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.3).normalize(),
            Vec3::new(0.4, -0.7, 0.5).normalize(),
        ] {
            let (e, no) = east_north(n);
            assert!((e.length() - 1.0).abs() < 1e-5);
            assert!((no.length() - 1.0).abs() < 1e-5);
            assert!(e.dot(n).abs() < 1e-5, "east not tangent");
            assert!(no.dot(n).abs() < 1e-5, "north not tangent");
            assert!(e.dot(no).abs() < 1e-5, "east·north not 0");
            assert!((e.cross(no) - n).length() < 1e-4, "frame not right-handed");
            // North points toward +Z.
            assert!(no.z > 0.0, "north should have a +Z component off the equator");
        }
    }

    /// A purely latitude-dependent field (T = a·sinφ) has a gradient that points due north with the
    /// analytically known slope a·cosφ / R per metre.
    #[test]
    fn gradient_matches_analytic_latitude_field() {
        let g = Grid::new(6000);
        let field: Vec<f32> = g.centers.iter().map(|c| 100.0 * c.z).collect(); // T = 100·sinφ
        let grad = gradients(&g, &field, R);
        for i in 0..g.n {
            let n = g.centers[i];
            let lat = n.z.asin();
            // Stay in mid-latitudes: near the poles the tangent frame is ill-defined and the
            // irregular neighbor spacing makes the discrete gradient noisy.
            if !(10.0..70.0).contains(&lat.abs().to_degrees()) {
                continue;
            }
            let (east, north) = east_north(n);
            let expected = 100.0 * lat.cos() / R; // d/d(arc) of 100·sin(arc)
            let got_north = grad[i].dot(north);
            let got_east = grad[i].dot(east);
            assert!(
                (got_north - expected).abs() < 0.15 * expected,
                "lat {:.1}°: north grad {got_north:.3e} vs {expected:.3e}",
                lat.to_degrees(),
            );
            // The field has no zonal variation, so the east component should be small relative to
            // the (purely meridional) gradient — only discretization noise remains.
            assert!(
                got_east.abs() < 0.15 * grad[i].length(),
                "lat {:.1}°: spurious east grad {got_east:.3e} vs |∇| {:.3e}",
                lat.to_degrees(),
                grad[i].length(),
            );
        }
    }

    /// The zonal-mean belts have the right sign in each cell: easterly (eastward < 0) in the
    /// trades and polar bands, westerly (> 0) in the mid-latitudes — in both hemispheres.
    #[test]
    fn belts_have_earthlike_sign_pattern() {
        let climate = Climate::default();
        for lat_deg in [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0f32] {
            let lat = lat_deg.to_radians();
            let n = Vec3::new(lat.cos(), 0.0, lat.sin());
            let (east, north) = east_north(n);
            let w = belt_wind(n, east, north, &climate, 1.0);
            let zonal = w.dot(east);
            let band = lat_deg.abs();
            if band < 30.0 || band > 60.0 {
                assert!(zonal < 0.0, "{lat_deg}° should be easterly, got {zonal}");
            } else {
                assert!(zonal > 0.0, "{lat_deg}° should be westerly, got {zonal}");
            }
        }
    }

    /// In the northern trade band the surface flow is north*easterly*: it blows toward the
    /// south-west (westward and equatorward), so it has both a negative eastward and a negative
    /// poleward (southward) component.
    #[test]
    fn northern_trades_are_northeasterly() {
        let climate = Climate::default();
        let lat = 15.0_f32.to_radians();
        let n = Vec3::new(lat.cos(), 0.0, lat.sin());
        let (east, north) = east_north(n);
        let w = belt_wind(n, east, north, &climate, 1.0);
        assert!(w.dot(east) < 0.0, "trade should blow westward");
        assert!(w.dot(north) < 0.0, "trade should blow equatorward (southward in N)");
    }

    /// The computed field is everywhere tangent to the sphere.
    #[test]
    fn wind_field_is_tangent() {
        let g = Grid::new(4000);
        let climate = Climate::default();
        // Equator-warm field so there is a real gradient to drive the perturbation.
        let temp: Vec<f32> = g.centers.iter().map(|c| 300.0 - 50.0 * c.z.abs()).collect();
        let grad = gradients(&g, &temp, R);
        let gfac = gradient_factor(&g, &temp);
        let mut wind = vec![Vec3::ZERO; g.n];
        compute(&g, &climate, gfac, &grad, &mut wind, 0, 2);
        for i in 0..g.n {
            assert!(wind[i].dot(g.centers[i]).abs() < 1e-3, "wind not tangent at {i}");
            assert!(wind[i].is_finite(), "non-finite wind at {i}");
        }
    }

    /// The geostrophic perturbation is a meaningful few m·s⁻¹ for a realistic temperature gradient
    /// (not the invisibly-small flow the old dimensionless coefficient produced), and stays finite
    /// straight across the equator where the Coriolis parameter vanishes.
    #[test]
    fn geostrophic_perturbation_is_significant_and_equator_safe() {
        let climate = Climate::default();
        let grad_mag = 5.0e-6; // ~ Earth's equator-to-pole gradient, K·m⁻¹

        // Mid-latitude: a northward gradient should drive a few-m·s⁻¹ wind.
        let lat = 35.0_f32.to_radians();
        let n = Vec3::new(lat.cos(), 0.0, lat.sin());
        let (_e, north) = east_north(n);
        let w = geostrophic_wind(n, north * grad_mag, &climate);
        assert!(
            (1.0..15.0).contains(&w.length()),
            "mid-lat geostrophic wind {} m/s out of expected few-m/s range",
            w.length(),
        );

        // Equator: f = 0, but friction keeps the balance finite (flow down-gradient).
        let n_eq = Vec3::new(1.0, 0.0, 0.0);
        let (_e2, north_eq) = east_north(n_eq);
        let w_eq = geostrophic_wind(n_eq, north_eq * grad_mag, &climate);
        assert!(w_eq.is_finite(), "equator wind non-finite");
        assert!(w_eq.length() > 0.1, "equator wind should be a real down-gradient flow");
    }

    /// A flat (uniform) temperature field gives no gradient factor and no geostrophic flow, but the
    /// belts still vanish because the gradient factor is 0 — a dead-calm planet.
    #[test]
    fn flat_climate_is_calm() {
        let g = Grid::new(2000);
        let climate = Climate::default();
        let temp = vec![288.0f32; g.n];
        let grad = gradients(&g, &temp, R);
        let gfac = gradient_factor(&g, &temp);
        let mut wind = vec![Vec3::ZERO; g.n];
        compute(&g, &climate, gfac, &grad, &mut wind, 0, 2);
        for w in &wind {
            assert!(w.length() < 1e-3, "uniform climate should be calm, got {}", w.length());
        }
    }

    /// The upper-layer belt is a westerly subtropical jet: eastward (westerly, > 0) at every
    /// latitude, peaking near 30° rather than alternating sign like the surface belts.
    #[test]
    fn upper_belt_is_westerly_jet_peaking_at_30() {
        let climate = Climate::default();
        let mut peak_lat = 0.0f32;
        let mut peak_u = 0.0f32;
        for lat_deg in (-85..=85).step_by(5) {
            let lat = (lat_deg as f32).to_radians();
            let n = Vec3::new(lat.cos(), 0.0, lat.sin());
            let (east, north) = east_north(n);
            let u = belt_wind_upper(n, east, north, &climate, 1.0).dot(east);
            assert!(u >= 0.0, "upper zonal flow should be westerly at {lat_deg}°, got {u}");
            if u > peak_u {
                peak_u = u;
                peak_lat = (lat_deg as f32).abs();
            }
        }
        assert!(peak_u > 1.0, "subtropical jet should be a real flow, got {peak_u} m/s");
        assert!(
            (peak_lat - 30.0).abs() <= 5.0,
            "subtropical jet should peak near 30°, peaked at {peak_lat}°",
        );
    }

    /// The upper-layer meridional (overturning return) branch is everywhere *opposite* the surface
    /// meridional branch — the signature of a closed overturning cell (e.g. equatorward at the
    /// surface under the trades, poleward aloft).
    #[test]
    fn upper_meridional_opposes_surface() {
        let climate = Climate::default();
        for lat_deg in [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0f32] {
            let lat = lat_deg.to_radians();
            let n = Vec3::new(lat.cos(), 0.0, lat.sin());
            let (east, north) = east_north(n);
            let surf = belt_wind(n, east, north, &climate, 1.0).dot(north);
            let upper = belt_wind_upper(n, east, north, &climate, 1.0).dot(north);
            assert!(
                surf * upper < 0.0,
                "{lat_deg}°: surface ({surf:.2}) and upper ({upper:.2}) meridional flow should oppose",
            );
        }
    }
}
