//! Hemisphere-symmetry diagnostic. Every other diagnostic bins by `|lat|`, folding the two
//! hemispheres together — so a north/south asymmetry is invisible to them. This one tracks the
//! area-weighted mean temperature of each hemisphere (and each model layer) over a multi-year run,
//! to catch a drift or symmetry-breaking instability.
//!
//! Run:
//!   cargo run --release --example hemisphere_diagnostics                 # 3 years, full model
//!   cargo run --release --example hemisphere_diagnostics -- 3 noseason   # obliquity = 0 (symmetric forcing)
//!   cargo run --release --example hemisphere_diagnostics -- 3 nocouple   # vertical_wind_coupling = 0
//!   cargo run --release --example hemisphere_diagnostics -- 3 nojet      # jet_wind_speed = 0
//!   cargo run --release --example hemisphere_diagnostics -- 3 noupperadv # no upper-layer advection
//!
//! `noseason` is the key test: with obliquity 0 the forcing is exactly N/S symmetric, so any
//! divergence is an internal (dynamical) symmetry break, not a seasonal artifact.

use std::io::Write;

use ecology_simulator::grid::Grid;
use ecology_simulator::sim::{Climate, Sim};

const K0: f32 = 273.15;

fn c(k: f32) -> f32 {
    k - K0
}

/// Area-weighted (north mean, south mean) of a per-cell field.
fn hemispheres(grid: &Grid, field: &[f32]) -> (f32, f32) {
    let (mut nt, mut na, mut st, mut sa) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..grid.n {
        let a = grid.areas[i] as f64;
        if grid.lonlat_deg[i].y >= 0.0 {
            nt += field[i] as f64 * a;
            na += a;
        } else {
            st += field[i] as f64 * a;
            sa += a;
        }
    }
    ((nt / na) as f32, (st / sa) as f32)
}

fn main() {
    let years: f32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let mode = std::env::args().nth(2).unwrap_or_else(|| "full".to_string());
    let cells: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8_000);

    let mut climate = Climate::default();
    match mode.as_str() {
        "noseason" => climate.obliquity = 0.0,
        "nocouple" => climate.vertical_wind_coupling = 0.0,
        "nojet" => climate.jet_wind_speed = 0.0,
        "noupperadv" => climate.upper_advection = false,
        "full" => {}
        other => {
            eprintln!("unknown mode '{other}'; using full");
        }
    }

    let grid = Grid::new(cells);
    let mut sim = Sim::new(grid.n, climate, 288.0);
    // Generate terrain so the ocean/land heat-capacity mask exists (oceans buffer the swing).
    sim.generate_terrain(&grid, 52, 0xC0FFEE);
    let day = climate.day_seconds;
    let total_days = (years * 365.25) as usize;
    let sample_every = 30usize;

    println!("=== Hemisphere diagnostics ===");
    println!("cells={}  years={years}  mode={mode}", grid.n);
    println!();
    println!("  day | surface N / S / N−S        | lower N / S          | upper N / S");
    println!("------+----------------------------+----------------------+--------------------");
    for d in 0..total_days {
        sim.advance(&grid, day);
        if d % sample_every == 0 {
            let (sn, ss) = hemispheres(&grid, sim.temperatures());
            let (ln, ls) = hemispheres(&grid, sim.lower_atmosphere_temperatures());
            let (un, us) = hemispheres(&grid, sim.upper_atmosphere_temperatures());
            println!(
                " {d:4} | {:5.1} / {:5.1} / {:+5.1} °C    | {:5.1} / {:5.1} °C    | {:5.1} / {:5.1} °C",
                c(sn), c(ss), sn - ss, c(ln), c(ls), c(un), c(us),
            );
            std::io::stdout().flush().ok();
        }
    }
}
