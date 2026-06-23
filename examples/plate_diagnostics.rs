//! Plate-size distribution diagnostics: measure the areas the terrain generator produces and
//! compare their inequality and shape to Earth's real tectonic plates, for both the Pareto and
//! lognormal size models.
//!
//! Run:
//!   cargo run --release --example plate_diagnostics
//!   cargo run --release --example plate_diagnostics -- 256   # seeds to average over
//!
//! Earth reference: the 52 plates of Bird (2003), PB2002 model, areas in steradians (whole
//! sphere = 4π ≈ 12.566 sr). The real distribution is strongly heavy-tailed — a few giant plates
//! and a long tail of microplates spanning four orders of magnitude.

use ecology_simulator::grid::Grid;
use ecology_simulator::sim::terrain::{SizeModel, Terrain};

const CELLS: usize = 16_384;

/// (name, area in steradians) for the 52 PB2002 plates, largest first.
const EARTH_PLATES: &[(&str, f64)] = &[
    ("Pacific", 2.57685),
    ("African", 1.44065),
    ("Antarctic", 1.43268),
    ("North American", 1.36559),
    ("Eurasian", 1.19630),
    ("Australian", 1.13294),
    ("South American", 1.03045),
    ("Somalia", 0.47192),
    ("Nazca", 0.39669),
    ("Indian", 0.30637),
    ("Sunda", 0.21967),
    ("Philippine Sea", 0.13409),
    ("Amur", 0.13066),
    ("Arabian", 0.12082),
    ("Okhotsk", 0.07482),
    ("Caribbean", 0.07304),
    ("Cocos", 0.07223),
    ("Yangtze", 0.05425),
    ("Scotia", 0.04190),
    ("Caroline", 0.03765),
    ("North Andes", 0.02394),
    ("Altiplano", 0.02050),
    ("Banda Sea", 0.01715),
    ("New Hebrides", 0.01585),
    ("Anatolian", 0.01418),
    ("Bird's Head", 0.01295),
    ("Burma", 0.01270),
    ("Kermadec", 0.01245),
    ("Woodlark", 0.01116),
    ("Mariana", 0.01037),
    ("Molucca Sea", 0.01030),
    ("North Bismarck", 0.00956),
    ("Timor", 0.00870),
    ("Okinawa", 0.00802),
    ("Aegean Sea", 0.00793),
    ("South Bismarck", 0.00762),
    ("Panama", 0.00674),
    ("Juan de Fuca", 0.00632),
    ("Tonga", 0.00625),
    ("Balmoral Reef", 0.00481),
    ("Sandwich", 0.00454),
    ("Easter", 0.00411),
    ("Conway Reef", 0.00356),
    ("Solomon Sea", 0.00317),
    ("Niuafo'ou", 0.00306),
    ("Maoke", 0.00284),
    ("Rivera", 0.00249),
    ("Juan Fernandez", 0.00241),
    ("Shetland", 0.00178),
    ("Futuna", 0.00079),
    ("Galapagos", 0.00036),
    ("Manus", 0.00020),
];

/// Gini coefficient of a set of areas (0 = all equal, →1 = maximally unequal).
fn gini(areas: &[f64]) -> f64 {
    let n = areas.len();
    if n == 0 {
        return 0.0;
    }
    let mut a = areas.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let sum: f64 = a.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let weighted: f64 = a.iter().enumerate().map(|(i, &x)| (i as f64 + 1.0) * x).sum();
    2.0 * weighted / (n as f64 * sum) - (n as f64 + 1.0) / n as f64
}

/// Coefficient of variation (std / mean) of a set of areas.
fn cv(areas: &[f64]) -> f64 {
    let n = areas.len() as f64;
    let mean = areas.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = areas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean
}

/// (Gini, CV, top-plate share %) for a set of areas.
fn metrics(areas: &[f64]) -> (f64, f64, f64) {
    let total: f64 = areas.iter().sum();
    let max = areas.iter().cloned().fold(0.0, f64::max);
    (gini(areas), cv(areas), max / total * 100.0)
}

/// Per-plate areas (steradians) for one generated terrain.
fn plate_areas(grid: &Grid, num_plates: usize, seed: u64, model: SizeModel) -> Vec<f64> {
    let terrain = Terrain::generate_with(grid, num_plates, seed, model);
    let mut areas = vec![0.0f64; terrain.plates.len()];
    for i in 0..grid.n {
        areas[terrain.plate_id[i] as usize] += grid.areas[i] as f64;
    }
    areas
}

/// Mean (Gini, CV, top share) over `seeds` generations.
fn mean_metrics(grid: &Grid, n: usize, seeds: u64, model: SizeModel) -> (f64, f64, f64) {
    let (mut g, mut c, mut t) = (0.0, 0.0, 0.0);
    for s in 0..seeds {
        let (mg, mc, mt) = metrics(&plate_areas(grid, n, s.wrapping_mul(0x9E37_79B9), model));
        g += mg;
        c += mc;
        t += mt;
    }
    let s = seeds as f64;
    (g / s, c / s, t / s)
}

/// Mean size profile: the average share (%) of the k-th largest plate over `seeds` generations.
fn mean_profile(grid: &Grid, n: usize, seeds: u64, model: SizeModel) -> Vec<f64> {
    let mut acc = vec![0.0f64; n];
    for s in 0..seeds {
        let mut a = plate_areas(grid, n, s.wrapping_mul(0x9E37_79B9), model);
        let total: f64 = a.iter().sum();
        a.sort_by(|x, y| y.partial_cmp(x).unwrap());
        for (k, &v) in a.iter().enumerate() {
            acc[k] += v / total * 100.0;
        }
    }
    acc.iter().map(|&v| v / seeds as f64).collect()
}

/// Bisection fit of a model parameter so the chosen `stat` (extracted from the (Gini, CV, top)
/// triple) ≈ `target`. `incr` says whether that stat increases with the parameter (lognormal σ)
/// or decreases (Pareto exponent), over the monotonic range `[lo0, hi0]`.
fn fit(
    grid: &Grid,
    n: usize,
    seeds: u64,
    target: f64,
    lo0: f32,
    hi0: f32,
    incr: bool,
    make: impl Fn(f32) -> SizeModel,
    stat: impl Fn(f64, f64, f64) -> f64,
) -> f32 {
    let (mut lo, mut hi) = (lo0, hi0);
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        let (g, c, t) = mean_metrics(grid, n, seeds, make(mid));
        let too_high = stat(g, c, t) > target;
        // Move the bound that keeps `target` bracketed, accounting for monotonicity direction.
        if too_high == incr {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Print a parameter sweep for one model at a fixed plate count, against the Earth reference.
fn sweep(grid: &Grid, n: usize, seeds: u64, label: &str, params: &[f32], make: impl Fn(f32) -> SizeModel) {
    println!("  {label} sweep (N = {n}):");
    println!("    {:>6} {:>6} {:>6} {:>8}", "param", "Gini", "CV", "top");
    for &p in params {
        let (g, c, t) = mean_metrics(grid, n, seeds, make(p));
        println!("    {:>6.2} {:>6.3} {:>6.2} {:>7.1}%", p, g, c, t);
    }
}

fn main() {
    let seeds: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let fit_seeds = seeds.min(96); // fewer seeds inside the bisection loop for speed

    let grid = Grid::new(CELLS);
    let cell_sr = 4.0 * std::f64::consts::PI / grid.n as f64;

    let mut earth: Vec<(&str, f64)> = EARTH_PLATES.to_vec();
    earth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let earth_area: Vec<f64> = earth.iter().map(|p| p.1).collect();

    println!("=== Plate-size distribution: lognormal vs. Pareto vs. Earth ===");
    println!(
        "grid {} cells (1 cell = {:.5} sr) · metrics averaged over {} seeds\n",
        grid.n, cell_sr, seeds,
    );
    println!(
        "Earth (Bird 2003, PB2002): {} plates, {:.4} sr total. Smallest plate (Manus, {:.5} sr) is",
        earth.len(),
        earth_area.iter().sum::<f64>(),
        earth_area.last().unwrap(),
    );
    println!(
        "below one grid cell, so the generator can't reproduce the extreme microplate tail.\n",
    );

    // --- Fit + compare both models at matched Gini, for N = 12 and the full N = 52. ---
    for &n in &[12usize, 52] {
        let earth_top = &earth_area[..n];
        let (eg, ec, et) = metrics(earth_top);

        // Fit each model so its mean CV matches Earth's, over the monotonic (pre-saturation)
        // range of each parameter. CV rises with σ and falls with the Pareto exponent.
        let cv_stat = |_g: f64, c: f64, _t: f64| c;
        let p_exp = fit(&grid, n, fit_seeds, ec, 0.5, 8.0, false, SizeModel::Pareto, cv_stat);
        let l_sig = fit(&grid, n, fit_seeds, ec, 0.1, 4.0, true, SizeModel::Lognormal, cv_stat);
        let (pg, pc, pt) = mean_metrics(&grid, n, seeds, SizeModel::Pareto(p_exp));
        let (lg, lc, lt) = mean_metrics(&grid, n, seeds, SizeModel::Lognormal(l_sig));

        println!("--- N = {n} plates (generator fitted to Earth's CV {ec:.2}) ---");
        println!("  {:<26} {:>6} {:>6} {:>11}", "", "Gini", "CV", "top plate");
        println!("  {:<26} {:>6.3} {:>6.2} {:>10.1}%", format!("Earth (top {n})"), eg, ec, et);
        println!("  {:<26} {:>6.3} {:>6.2} {:>10.1}%", format!("Pareto(exp {p_exp:.2})"), pg, pc, pt);
        println!("  {:<26} {:>6.3} {:>6.2} {:>10.1}%", format!("Lognormal(σ {l_sig:.2})"), lg, lc, lt);
        // Closeness on the two metrics NOT used for fitting (Gini and top share).
        let perr = (pg - eg).abs() / eg + (pt - et).abs() / et;
        let lerr = (lg - eg).abs() / eg + (lt - et).abs() / et;
        println!(
            "  → shape error (Gini+top, lower=better):  Pareto {:.2}   Lognormal {:.2}   winner: {}\n",
            perr,
            lerr,
            if lerr < perr { "Lognormal" } else { "Pareto" },
        );

        // For the full catalogue, also show how Gini saturates with the model parameter — the
        // resolution ceiling (smallest plate ≥ 1 cell) caps how unequal the generator can get.
        if n == 52 {
            sweep(&grid, n, seeds, "Lognormal σ", &[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], SizeModel::Lognormal);
            sweep(&grid, n, seeds, "Pareto exp", &[1.0, 0.7, 0.5, 0.4, 0.3, 0.25], SizeModel::Pareto);
            println!();
        }
    }

    // --- Visual size profile at N = 12 (the simulator's plate count). ---
    let n = 12;
    let earth_top = &earth_area[..n];
    let earth_total: f64 = earth_top.iter().sum();
    let cv_stat = |_g: f64, c: f64, _t: f64| c;
    let target_cv = metrics(earth_top).1;
    let p_exp = fit(&grid, n, fit_seeds, target_cv, 0.5, 8.0, false, SizeModel::Pareto, cv_stat);
    let l_sig = fit(&grid, n, fit_seeds, target_cv, 0.1, 4.0, true, SizeModel::Lognormal, cv_stat);
    let p_prof = mean_profile(&grid, n, seeds, SizeModel::Pareto(p_exp));
    let l_prof = mean_profile(&grid, n, seeds, SizeModel::Lognormal(l_sig));

    println!("--- Mean size profile, N = 12 (share of k-th largest plate, %) ---");
    println!("  {:>4} {:>8} {:>8} {:>10}", "rank", "Earth", "Pareto", "Lognormal");
    for k in 0..n {
        println!(
            "  {:>4} {:>7.1}% {:>7.1}% {:>9.1}%",
            k + 1,
            earth_top[k] / earth_total * 100.0,
            p_prof[k],
            l_prof[k],
        );
    }
}
