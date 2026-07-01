//! Headless wind diagnostics: spin the climate up with and without atmospheric transport, then
//! report (a) the per-latitude-band zonal-mean surface wind and speed — to check the tri-cell
//! belt structure (easterly trades / westerly mid-latitudes / polar easterlies) — and (b) the
//! equator-to-pole temperature contrast *before vs. after* coupling, to confirm the winds + eddy
//! transport actually warm the poles.
//!
//! Run:
//!   cargo run --release --example wind_diagnostics
//!   cargo run --release --example wind_diagnostics -- 60   # override spin-up days
//!
//! The positional arg sets the spin-up length in days; everything else uses production defaults.

use ecology_simulator::grid::Grid;
use ecology_simulator::sim::wind::east_north;
use ecology_simulator::sim::{Climate, Sim};

const CELLS: usize = 16_384;
const K0: f32 = 273.15; // 0 °C in kelvin
const BANDS: usize = 9; // 10°-wide |lat| bands: 0–10, 10–20, … 80–90

fn c(kelvin: f32) -> f32 {
    kelvin - K0
}

/// Spin a sim up `days` days in hourly chunks to forget the initial condition.
fn spin_up(sim: &mut Sim, grid: &Grid, days: f32) {
    let chunk = sim.climate.day_seconds / 24.0;
    for _ in 0..(days * 24.0).round() as usize {
        sim.advance(grid, chunk);
    }
}

/// Area-weighted equator (|lat|<15°) minus pole (|lat|>75°) mean temperature (K).
fn equator_pole_contrast(grid: &Grid, temp: &[f32]) -> (f32, f32) {
    let (mut et, mut ea, mut pt, mut pa) = (0.0f64, 0.0, 0.0, 0.0);
    for i in 0..grid.n {
        let lat = grid.lonlat_deg[i].y.abs();
        let a = grid.areas[i] as f64;
        if lat < 15.0 {
            et += temp[i] as f64 * a;
            ea += a;
        } else if lat > 75.0 {
            pt += temp[i] as f64 * a;
            pa += a;
        }
    }
    ((et / ea) as f32, (pt / pa) as f32)
}

fn main() {
    let spin_up_days: f32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40.0);

    let grid = Grid::new(CELLS);

    // --- Baseline: original bare-surface model, all atmospheric transport off. ---
    let mut calm = Climate::default();
    calm.trade_wind_speed = 0.0;
    calm.meridional_wind_speed = 0.0;
    calm.geostrophic_coeff = 0.0;
    calm.eddy_diffusivity = 0.0;
    let mut base_sim = Sim::new(grid.n, calm, 288.0);
    spin_up(&mut base_sim, &grid, spin_up_days);
    let (base_eq, base_pole) = equator_pole_contrast(&grid, base_sim.temperatures());

    // --- Coupled: winds + storm-track eddy transport (production defaults). ---
    let mut sim = Sim::new(grid.n, Climate::default(), 288.0);
    spin_up(&mut sim, &grid, spin_up_days);
    let (eq, pole) = equator_pole_contrast(&grid, sim.temperatures());

    // --- Per-band zonal-mean wind (east component) and mean speed, area-weighted. ---
    let mut zonal = [0.0f64; BANDS];
    let mut speed = [0.0f64; BANDS];
    let mut band_area = [0.0f64; BANDS];
    let winds = sim.winds();
    for i in 0..grid.n {
        let lat = grid.lonlat_deg[i].y.abs();
        let b = ((lat / 10.0) as usize).min(BANDS - 1);
        let a = grid.areas[i] as f64;
        let (east, _north) = east_north(grid.centers[i]);
        zonal[b] += winds[i].dot(east) as f64 * a;
        speed[b] += winds[i].length() as f64 * a;
        band_area[b] += a;
    }

    println!("=== Wind diagnostics ===");
    println!("cells={CELLS}  spin-up={spin_up_days} days");
    println!();
    println!("Equator-to-pole temperature contrast (pole warming = transport working):");
    println!(
        "  no transport : equator {:5.1} °C  pole {:6.1} °C   ΔT = {:5.1} °C",
        c(base_eq),
        c(base_pole),
        base_eq - base_pole,
    );
    println!(
        "  with winds   : equator {:5.1} °C  pole {:6.1} °C   ΔT = {:5.1} °C",
        c(eq),
        c(pole),
        eq - pole,
    );
    println!(
        "  → poles {:+.1} °C, equator {:+.1} °C, gradient {:+.1} °C  (Earth ΔT ≈ 50 °C)",
        c(pole) - c(base_pole),
        c(eq) - c(base_eq),
        (eq - pole) - (base_eq - base_pole),
    );
    println!();
    println!(" |lat| band | zonal-mean wind          | mean speed");
    println!("-----------+--------------------------+-----------");
    for b in 0..BANDS {
        if band_area[b] <= 0.0 {
            continue;
        }
        let inv = 1.0 / band_area[b];
        let u = (zonal[b] * inv) as f32;
        let dir = if u >= 0.0 { "westerly →" } else { "easterly ←" };
        println!(
            "  {:2}–{:2}°   | {:+5.1} m/s  {:10} | {:5.1} m/s",
            b * 10,
            b * 10 + 10,
            u,
            dir,
            (speed[b] * inv) as f32,
        );
    }
    println!();
    println!("Expected Earth pattern: easterly 0–30° (trades), westerly 30–60°, easterly 60–90°.");

    // --- Upper layer: subtropical jet (zonal) + overturning closure (meridional vs surface). ---
    // The upper-level zonal mean should show a westerly jet peaking near 30°, and the upper
    // meridional flow should be *opposite* the surface in every band (closed overturning cells).
    let mut up_zonal = [0.0f64; BANDS];
    let mut up_merid = [0.0f64; BANDS];
    let mut sf_merid = [0.0f64; BANDS];
    let winds_hi = sim.winds_hi();
    for i in 0..grid.n {
        let lat = grid.lonlat_deg[i].y.abs();
        let b = ((lat / 10.0) as usize).min(BANDS - 1);
        let a = grid.areas[i] as f64;
        let (east, north) = east_north(grid.centers[i]);
        up_zonal[b] += winds_hi[i].dot(east) as f64 * a;
        up_merid[b] += winds_hi[i].dot(north) as f64 * a;
        sf_merid[b] += winds[i].dot(north) as f64 * a;
    }
    println!();
    println!(" |lat| band | upper zonal (jet)        | surface merid | upper merid | overturning");
    println!("-----------+--------------------------+---------------+-------------+------------");
    for b in 0..BANDS {
        if band_area[b] <= 0.0 {
            continue;
        }
        let inv = 1.0 / band_area[b];
        let uz = (up_zonal[b] * inv) as f32;
        let sm = (sf_merid[b] * inv) as f32;
        let um = (up_merid[b] * inv) as f32;
        let dir = if uz >= 0.0 { "westerly →" } else { "easterly ←" };
        // Opposed signs = a closed cell (surface and upper branches return each other's mass).
        let closed = if sm * um < 0.0 { "✓ opposed" } else { "— same" };
        println!(
            "  {:2}–{:2}°   | {:+5.1} m/s  {:10} | {:+6.1} m/s   | {:+6.1} m/s | {}",
            b * 10,
            b * 10 + 10,
            uz,
            dir,
            sm,
            um,
            closed,
        );
    }
    println!();
    println!("Expected: westerly subtropical jet peaking ~30°; upper meridional opposite the surface.");

    // --- Temporal variation: do the winds evolve as the day/night warm spot moves? Measured for
    // both layers — the upper layer's variation is how "lively" the magenta arrows look. ---
    let surf_before: Vec<glam::Vec3> = sim.winds().to_vec();
    let upper_before: Vec<glam::Vec3> = sim.winds_hi().to_vec();
    let day = sim.climate.day_seconds;
    // advance a quarter day in hourly steps (a full day would return the sun — and the periodic
    // diurnal wind — to its start, masking the within-day change). advance re-diagnoses the wind.
    for _ in 0..6 {
        sim.advance(&grid, day / 24.0);
    }
    let variation = |before: &[glam::Vec3], after: &[glam::Vec3]| -> (f64, f64) {
        let (mut dsum, mut bsum, mut area) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..grid.n {
            let a = grid.areas[i] as f64;
            dsum += (after[i] - before[i]).length() as f64 * a;
            bsum += before[i].length() as f64 * a;
            area += a;
        }
        (dsum / area, bsum / area)
    };
    let (s_change, s_speed) = variation(&surf_before, sim.winds());
    let (u_change, u_speed) = variation(&upper_before, sim.winds_hi());
    println!();
    println!(
        "Temporal variation over ¼ day (turbulence = lively arrows):\n  \
         surface: mean |Δwind| = {:.2} m/s  ({:.0}% of mean speed {:.2} m/s)\n  \
         upper:   mean |Δwind| = {:.2} m/s  ({:.0}% of mean speed {:.2} m/s)",
        s_change,
        100.0 * s_change / s_speed,
        s_speed,
        u_change,
        100.0 * u_change / u_speed,
        u_speed,
    );
}
