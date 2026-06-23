//! Terrain / hypsometry diagnostics: measure the elevation field the generator produces and
//! compare its gross statistics and shape to Earth's.
//!
//! The generator is built from physical quantities (crust thickness/density, isostasy, a fixed
//! ocean volume), so these numbers are *emergent* — the point of this tool is to check that the
//! physics lands near Earth and to calibrate the few free knobs ([`Params`]) when it doesn't.
//!
//! Run:
//!   cargo run --release --example terrain_diagnostics
//!   cargo run --release --example terrain_diagnostics -- 32   # seeds to average scalars over
//!
//! Earth reference (ETOPO-class global hypsometry):
//!   land fraction      29.2 %
//!   mean land height   +840 m
//!   mean ocean depth   −3682 m
//!   highest / lowest   +8848 m / −10994 m
//!   continental crust  ~41 % of surface area (≈ land + submerged shelves)

use ecology_simulator::grid::Grid;
use ecology_simulator::sim::elevation::{Crust, Params};
use ecology_simulator::sim::terrain::Terrain;

const CELLS: usize = 16_384;
const NUM_PLATES: usize = 52;

/// Area-weighted mean of `values` over the cells selected by `keep`.
fn mean_where(values: &[f32], areas: &[f32], keep: impl Fn(f32) -> bool) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..values.len() {
        if keep(values[i]) {
            num += values[i] as f64 * areas[i] as f64;
            den += areas[i] as f64;
        }
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        0.0
    }
}

/// Area fraction of the surface for which `keep(elevation)` holds.
fn area_fraction(values: &[f32], areas: &[f32], keep: impl Fn(f32) -> bool) -> f32 {
    let total: f64 = areas.iter().map(|&a| a as f64).sum();
    let sel: f64 = (0..values.len())
        .filter(|&i| keep(values[i]))
        .map(|i| areas[i] as f64)
        .sum();
    (sel / total) as f32
}

/// Elevation at each cumulative-area fraction measured from the highest point downward (the
/// hypsographic curve): for `f` in `fracs`, the elevation `e` such that the area above `e` is `f`.
fn hypsographic(elev: &[f32], areas: &[f32], fracs: &[f64]) -> Vec<f32> {
    let total: f64 = areas.iter().map(|&a| a as f64).sum();
    let mut idx: Vec<usize> = (0..elev.len()).collect();
    idx.sort_by(|&a, &b| elev[b].total_cmp(&elev[a])); // descending elevation
    let mut out = Vec::with_capacity(fracs.len());
    let mut acc = 0.0f64;
    let mut k = 0usize;
    let mut fi = 0usize;
    while fi < fracs.len() {
        let target = fracs[fi] * total;
        while k < idx.len() && acc < target {
            acc += areas[idx[k]] as f64;
            k += 1;
        }
        let j = k.saturating_sub(1).min(idx.len() - 1);
        out.push(elev[idx[j]]);
        fi += 1;
    }
    out
}

fn main() {
    let seeds: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let grid = Grid::new(CELLS);
    let params = Params::default();

    println!("=== Terrain / hypsometry diagnostics ===");
    println!(
        "grid {} cells · {} plates · {} seeds · water budget {:.0} m·sr (Earth) · cont. target {:.0}%\n",
        grid.n,
        NUM_PLATES,
        seeds,
        params.water_volume,
        params.continental_area * 100.0,
    );

    // --- Scalar statistics averaged over seeds, against Earth. ---
    let (mut land_frac, mut land_h, mut ocean_d, mut cont_area) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for s in 0..seeds {
        let seed = s.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        let terrain = Terrain::generate(&grid, NUM_PLATES, seed);
        let c = Crust::generate_with(&grid, &terrain, seed, params);
        land_frac += c.land_fraction as f64;
        land_h += mean_where(&c.elevation, &grid.areas, |e| e > 0.0) as f64;
        ocean_d += mean_where(&c.elevation, &grid.areas, |e| e <= 0.0) as f64;
        cont_area +=
            area_fraction(&c.continental_fraction, &grid.areas, |phi| phi > 0.5) as f64;
        hi += c.elevation.iter().cloned().fold(f32::MIN, f32::max) as f64;
        lo += c.elevation.iter().cloned().fold(f32::MAX, f32::min) as f64;
    }
    let n = seeds as f64;
    println!("  {:<22} {:>12} {:>12}", "", "generated", "Earth");
    println!("  {:<22} {:>11.1}% {:>11.1}%", "land fraction", land_frac / n * 100.0, 29.2);
    println!("  {:<22} {:>11.0}m {:>11.0}m", "mean land height", land_h / n, 840.0);
    println!("  {:<22} {:>11.0}m {:>11.0}m", "mean ocean depth", ocean_d / n, -3682.0);
    println!("  {:<22} {:>11.0}m {:>11.0}m", "highest", hi / n, 8848.0);
    println!("  {:<22} {:>11.0}m {:>11.0}m", "lowest", lo / n, -10994.0);
    println!("  {:<22} {:>11.1}% {:>11.1}%", "continental crust area", cont_area / n * 100.0, 41.0);

    // --- Hypsographic curve + ASCII hypsogram for one representative seed. ---
    let seed = 0x5EED_u64;
    let terrain = Terrain::generate(&grid, NUM_PLATES, seed);
    let c = Crust::generate_with(&grid, &terrain, seed, params);

    let fracs: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
    let curve = hypsographic(&c.elevation, &grid.areas, &fracs);
    println!("\n--- Hypsographic curve (seed {seed:#x}): elevation vs. area fraction above it ---");
    println!("  {:>8} {:>10}   {}", "area↑", "elev (m)", "profile (− ocean | + land)");
    let span = 8000.0f32; // m to either side of sea level for the bar
    for (f, &e) in fracs.iter().zip(curve.iter()) {
        // Bar centered at sea level: '#' left of center for depth, right for height.
        let half = 30usize;
        let pos = ((e / span).clamp(-1.0, 1.0) * half as f32) as i32;
        let mut bar = vec![b' '; half * 2 + 1];
        bar[half] = b'|';
        if pos >= 0 {
            for x in 0..pos {
                bar[half + 1 + x as usize] = b'#';
            }
        } else {
            for x in 0..(-pos) {
                bar[half - 1 - x as usize] = b'#';
            }
        }
        println!(
            "  {:>7.0}% {:>10.0}   {}",
            f * 100.0,
            e,
            String::from_utf8(bar).unwrap()
        );
    }

    // --- Boundary-class census (sanity check on the tectonic classifier). ---
    use ecology_simulator::sim::elevation::BoundaryKind::*;
    let count = |k| c.boundary.iter().filter(|&&b| b == k).count();
    let bn = c.boundary.len() as f32;
    println!("\n--- Plate-boundary cells (seed {seed:#x}) ---");
    println!(
        "  convergent {:.1}%   divergent {:.1}%   transform {:.1}%   interior {:.1}%",
        count(Convergent) as f32 / bn * 100.0,
        count(Divergent) as f32 / bn * 100.0,
        count(Transform) as f32 / bn * 100.0,
        count(None) as f32 / bn * 100.0,
    );
}
