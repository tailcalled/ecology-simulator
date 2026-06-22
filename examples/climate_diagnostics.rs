//! Headless climate diagnostics: run the temperature simulation to a quasi-steady diurnal
//! cycle and report, per latitude band, the mean temperature and the day/night swing. This
//! measures exactly what hovering the cursor only estimates roughly.
//!
//! Run:
//!   cargo run --release --example climate_diagnostics
//!   cargo run --release --example climate_diagnostics -- 2.0e6   # override heat capacity
//!   cargo run --release --example climate_diagnostics -- 2.0e6 60   # ...and spin-up days
//!
//! The single positional arg overrides `Climate::heat_capacity` (J·m⁻²·K⁻¹); the second sets
//! the spin-up length in days. Everything else uses the production defaults.

use ecology_simulator::grid::Grid;
use ecology_simulator::sim::{Climate, Sim, STEFAN_BOLTZMANN};

const CELLS: usize = 16_384;
const K0: f32 = 273.15; // 0 °C in kelvin

fn c(kelvin: f32) -> f32 {
    kelvin - K0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut climate = Climate::default();
    if let Some(hc) = args.next().and_then(|s| s.parse::<f32>().ok()) {
        climate.heat_capacity = hc;
    }
    let spin_up_days: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(40.0);

    let grid = Grid::new(CELLS);
    let mut sim = Sim::new(grid.n, climate, 288.0);

    let day = climate.day_seconds;

    // --- Spin up to forget the initial condition (thermal memory grows with heat capacity). ---
    let spin_chunk = day / 24.0; // hourly chunks
    let spin_chunks = (spin_up_days * 24.0).round() as usize;
    for _ in 0..spin_chunks {
        sim.advance(&grid, spin_chunk);
    }

    // --- Sample one full day, tracking per-cell min / max / mean temperature. ---
    const SAMPLES: usize = 144; // every 10 minutes of sim time
    let mut tmin = vec![f32::INFINITY; grid.n];
    let mut tmax = vec![f32::NEG_INFINITY; grid.n];
    let mut tsum = vec![0.0f64; grid.n];
    let sample_dt = day / SAMPLES as f32;
    for _ in 0..SAMPLES {
        sim.advance(&grid, sample_dt);
        for (i, &t) in sim.temperatures().iter().enumerate() {
            tmin[i] = tmin[i].min(t);
            tmax[i] = tmax[i].max(t);
            tsum[i] += t as f64;
        }
    }
    let tmean: Vec<f32> = tsum.iter().map(|&s| (s / SAMPLES as f64) as f32).collect();

    // --- Bin cells into 10°-wide absolute-latitude bands. ---
    const BANDS: usize = 9; // 0–10, 10–20, … 80–90
    let mut band_mean = [0.0f64; BANDS];
    let mut band_min = [0.0f64; BANDS];
    let mut band_max = [0.0f64; BANDS];
    let mut band_range = [0.0f64; BANDS];
    let mut band_area = [0.0f64; BANDS];
    let mut global_t = 0.0f64;
    let mut global_area = 0.0f64;
    for i in 0..grid.n {
        let lat = grid.lonlat_deg[i].y.abs();
        let b = ((lat / 10.0) as usize).min(BANDS - 1);
        let a = grid.areas[i] as f64;
        band_mean[b] += tmean[i] as f64 * a;
        band_min[b] += tmin[i] as f64 * a;
        band_max[b] += tmax[i] as f64 * a;
        band_range[b] += (tmax[i] - tmin[i]) as f64 * a;
        band_area[b] += a;
        global_t += tmean[i] as f64 * a;
        global_area += a;
    }

    // --- Reference: bare radiative-equilibrium global mean, εσT⁴ = S(1−α)/4. ---
    let absorbed = climate.solar_constant * (1.0 - climate.albedo) / 4.0;
    let eq = (absorbed / (climate.emissivity * STEFAN_BOLTZMANN)).powf(0.25);

    println!("=== Climate diagnostics ===");
    println!(
        "cells={}  heat_capacity={:.3e} J·m⁻²·K⁻¹  albedo={}  emissivity={}  spin-up={} days",
        grid.n, climate.heat_capacity, climate.albedo, climate.emissivity, spin_up_days,
    );
    println!(
        "global mean (area-weighted): {:.1} K  ({:.1} °C)   [radiative-equilibrium ref {:.1} K / {:.1} °C]",
        global_t / global_area,
        c((global_t / global_area) as f32),
        eq,
        c(eq),
    );
    println!();
    println!(" |lat| band |   mean   |  night low |  day high  | diurnal range");
    println!("-----------+----------+------------+------------+--------------");
    for b in 0..BANDS {
        if band_area[b] <= 0.0 {
            continue;
        }
        let inv = 1.0 / band_area[b];
        println!(
            "  {:2}–{:2}°   | {:5.1} °C | {:6.1} °C  | {:6.1} °C  |  {:5.1} °C",
            b * 10,
            b * 10 + 10,
            c((band_mean[b] * inv) as f32),
            c((band_min[b] * inv) as f32),
            c((band_max[b] * inv) as f32),
            (band_range[b] * inv) as f32,
        );
    }
}
