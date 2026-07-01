//! Climate simulation: the day/night cycle and surface temperature.
//!
//! The planet is the unit sphere with +Z as the north pole. Rather than rotating the mesh,
//! we move the **subsolar point** (the direction to the sun) through the planet's fixed
//! frame: its longitude sweeps once per day, and its latitude (solar declination) oscillates
//! once per year with the axial tilt. Each cell then exchanges energy by absorbing sunlight,
//! radiating heat (Stefan–Boltzmann), and conducting laterally to its neighbors.
//!
//! The per-cell update is data-parallel (rayon), reading a stable snapshot and writing a
//! scratch buffer (double buffering), so neighbor reads never race writes.

pub mod elevation;
mod temperature;
pub mod terrain;
pub mod wind;

use glam::Vec3;

use crate::grid::Grid;
use elevation::Crust;
use terrain::Terrain;

/// Stefan–Boltzmann constant (W·m⁻²·K⁻⁴).
pub const STEFAN_BOLTZMANN: f32 = 5.670_374_4e-8;

/// Tunable climate parameters. Defaults are roughly Earth-like and produce a stable
/// day/night thermal lag with an equator-warm / pole-cold gradient.
#[derive(Clone, Copy, Debug)]
pub struct Climate {
    /// Peak solar flux at the subsolar point (W·m⁻²).
    pub solar_constant: f32,
    /// Fraction of incoming sunlight reflected (0..1).
    pub albedo: f32,
    /// Thermal emissivity for outgoing radiation (0..1).
    pub emissivity: f32,
    /// Areal heat capacity of **land** surface (J·m⁻²·K⁻¹) — a thin thermally-active skin, so it
    /// responds fast (a clear diurnal + seasonal swing, as continents have). Applied to cells with
    /// elevation ≥ 0; ocean cells use [`heat_capacity_ocean`] instead.
    pub heat_capacity: f32,
    /// Areal heat capacity of **ocean** surface (J·m⁻²·K⁻¹) — the wind-mixed layer (~70 m of water),
    /// ~500× land. Its huge inertia keeps the sea-surface temperature nearly flat through the day and
    /// the seasons, which (oceans being ~70% of the surface) is what holds each hemisphere's mean
    /// temperature steady instead of swinging ±50 °C with the season. Applied to cells with elevation < 0.
    pub heat_capacity_ocean: f32,
    /// Lateral heat-conduction coefficient (W·m⁻²·K⁻¹ per unit conductance).
    pub diffusivity: f32,
    /// Axial tilt (radians); drives seasonal variation of the subsolar latitude.
    pub obliquity: f32,
    /// Length of one day in simulation seconds.
    pub day_seconds: f32,
    /// Length of one year in simulation seconds.
    pub year_seconds: f32,

    // --- Atmospheric transport (winds) ---
    /// Planet radius (m). Turns the unit-sphere arc lengths into physical distances, needed for
    /// temperature gradients (K·m⁻¹) and so wind speeds in m·s⁻¹.
    pub planet_radius: f32,
    /// Nominal peak speed of the zonal-mean belts (trades / westerlies / polar easterlies), m·s⁻¹,
    /// at the reference equator-to-pole temperature contrast. Scales with the simulated contrast.
    pub trade_wind_speed: f32,
    /// Nominal peak speed of the meridional (poleward/equatorward) surface branch of the
    /// overturning cells, m·s⁻¹. Smaller than the zonal speed, as on Earth. Reused (with reversed
    /// sign) for the upper layer's overturning return branch — mass continuity makes the two
    /// branches comparable in a single column.
    pub meridional_wind_speed: f32,
    /// Nominal peak speed of the upper-tropospheric **subtropical jet** (m·s⁻¹) at the reference
    /// gradient — the westerly maximum at ~30° aloft. Earth's runs ~25–40 m·s⁻¹.
    pub jet_wind_speed: f32,
    /// Pressure-gradient acceleration per unit temperature gradient (m²·s⁻²·K⁻¹) — i.e.
    /// `(dp/dT)/ρ` — converting the temperature gradient into the pressure-gradient force that
    /// drives the geostrophic perturbation (the monsoon / continental-high departures, and the
    /// only time-varying part of the wind as the day/night warm spot moves).
    pub geostrophic_coeff: f32,
    /// Linear surface-friction (Ekman drag) coefficient (s⁻¹) used in the diagnostic *seed* wind.
    /// Turns the near-surface wind across the isobars toward low pressure; keeps the balance finite
    /// at the equator. 0 = pure geostrophic (flow along isobars, singular at the equator).
    pub wind_friction: f32,
    /// Relaxation rate (s⁻¹) pulling the prognostic wind toward the analytic tri-cell belts. This
    /// is what *maintains* the Hadley/Ferrel/Polar structure (which the dynamics can't generate on
    /// their own) and also acts as the surface drag. Larger = belts dominate; smaller = freer eddies.
    pub wind_relax: f32,
    /// Eddy viscosity (s⁻¹, applied to the conductance-weighted velocity Laplacian) damping the
    /// smallest scales so the central momentum advection stays numerically stable.
    pub wind_viscosity: f32,
    /// Rayleigh drag (s⁻¹) on the **upper** wind layer. Much weaker than the surface
    /// [`wind_friction`] — the free troposphere is far from the surface boundary layer — so the
    /// subtropical jet can run fast. Still nonzero for eddy dissipation and the equatorial bound.
    pub wind_friction_upper: f32,
    /// Vertical momentum coupling (s⁻¹): rate at which the upper wind is relaxed toward the surface
    /// wind, representing vertical mixing / the barotropic component (thermal wind: `u_upper ≈
    /// u_lower + shear`). Without it the upper layer, forced only by the slow upper-temperature
    /// field, is nearly steady; this injects the turbulent surface wind's variability aloft so the
    /// jet is lively. Kept below `wind_relax` so the jet's belt shear still survives.
    pub vertical_wind_coupling: f32,
    /// Extra lateral heat conductance representing mid-latitude baroclinic-eddy heat transport,
    /// added to `diffusivity` and peaking in the storm-track band. This — not surface advection —
    /// is what carries heat poleward and warms the poles. Same units as `diffusivity`.
    pub eddy_diffusivity: f32,

    // --- Atmosphere (TWO grey-radiation layers — lower + upper — + a convection chain) ---
    /// IR emissivity/absorptivity ε_L (0..1) of the **lower** atmosphere layer: the fraction of the
    /// longwave beam it absorbs (= the fraction it emits, by Kirchhoff). The two layers together set
    /// the greenhouse strength and, because the upper layer is colder, an emergent vertical
    /// temperature structure.
    pub atm_emissivity_lower: f32,
    /// IR emissivity/absorptivity ε_U (0..1) of the **upper** atmosphere layer. Smaller than the
    /// lower (less mass/absorber aloft), so the upper layer is optically thinner.
    pub atm_emissivity_upper: f32,
    /// Areal heat capacity C_L (J·m⁻²·K⁻¹) of the lower layer — its share of the column thermal mass.
    pub atm_heat_capacity_lower: f32,
    /// Areal heat capacity C_U (J·m⁻²·K⁻¹) of the upper layer. `C_L + C_U` ≈ the old single-slab
    /// column mass (~1e7).
    pub atm_heat_capacity_upper: f32,
    /// Convective (sensible-heat) exchange coefficient (W·m⁻²·K⁻¹): the upward heat flux per degree
    /// of super-threshold warmth across a convective link. Used for both links of the chain
    /// (surface→lower and lower→upper) — hot air carries heat up the column instead of the surface
    /// running away.
    pub convection_coeff: f32,
    /// Convective lapse threshold Γ (K) *per link*: a link fires only when the lower body is warmer
    /// than the upper by more than Γ, so each layer settles ~Γ below the one beneath it (surface →
    /// lower → upper). This keeps each layer colder than the one below — which is what makes the
    /// greenhouse work — and builds the vertical lapse. Applied to both links of the chain.
    pub convection_threshold: f32,
    /// Ablation switch (diagnostics): when false, the upper layer `T_u` is not advected by the upper
    /// wind. Always true in production; lets a diagnostic isolate the upper-advection channel.
    pub upper_advection: bool,
    /// Lateral heat conductance within the atmosphere (same units as `diffusivity`); the atmosphere
    /// mixes heat horizontally more freely than the surface.
    pub atm_diffusivity: f32,
}

impl Default for Climate {
    fn default() -> Self {
        Self {
            // Earth's solar constant (total solar irradiance at 1 AU), W·m⁻².
            solar_constant: 1361.0,
            // Earth's mean Bond albedo (dimensionless).
            albedo: 0.3,
            // Surface now radiates as a near-blackbody; the greenhouse is modelled explicitly by the
            // atmosphere layer (`atm_emissivity`), so the old 0.61 fudge is gone. Dimensionless.
            emissivity: 1.0,
            // Land: ~0.75-day thermal time constant — clearly visible diurnal swing with a realistic
            // afternoon-warm / pre-dawn-cold lag. Areal heat capacity, J·m⁻²·K⁻¹.
            heat_capacity: 4.0e5,
            // Ocean mixed layer (~70 m of water): ρ·c_p·depth ≈ 1025·4000·70. ~700× land, so the sea
            // surface barely changes through day or season — damps the hemispheric seasonal swing.
            heat_capacity_ocean: 2.9e8,
            // Lateral heat-conduction coefficient, W·m⁻²·K⁻¹ per unit conductance.
            diffusivity: 25.0,
            // Earth's axial tilt, radians.
            obliquity: 23.44_f32.to_radians(),
            // Earth's solar day, seconds.
            day_seconds: 86_400.0,
            // Earth's year (365.25 days), seconds.
            year_seconds: 86_400.0 * 365.25,

            // Earth's mean radius, metres.
            planet_radius: 6.371e6,
            // Peak surface trade/westerly speed (~7–10 m·s⁻¹ on Earth).
            trade_wind_speed: 9.0,
            // Meridional surface branch is weaker than the zonal flow.
            meridional_wind_speed: 3.0,
            // Subtropical jet peak (~30 m·s⁻¹), the upper-level westerly maximum at 30°.
            jet_wind_speed: 30.0,
            // (dp/dT)/ρ ≈ (67 Pa·K⁻¹)/(1.2 kg·m⁻³): gives a few-m·s⁻¹ geostrophic perturbation for
            // typical gradients, so weather-scale flow is visible alongside the steady belts.
            geostrophic_coeff: 60.0,
            // Light Rayleigh drag (~1.4-day timescale): dissipates eddies and bounds the wind near
            // the equator, without flattening the belts (which the zonal-mean relaxation maintains).
            wind_friction: 8.0e-6,
            // Zonal-mean relaxation (~9-hour timescale): strongly holds the band-mean wind to the
            // tri-cell belts. Acts only on the mean, so it preserves the cells without damping eddies.
            wind_relax: 3.0e-5,
            // Eddy viscosity sized against the momentum CFL (ν·maxΣw·dt well under 1) — small, so
            // turbulence survives down toward the grid scale.
            wind_viscosity: 2.0e-4,
            // Upper-layer Rayleigh drag, equal to the surface value (the free troposphere is
            // weakly damped). This lets the upper eddies turbulence freely — lively upper-wind
            // arrows — which the *two*-layer thermodynamics can afford: with a cold upper layer
            // advected separately from the lower, the overturning transports heat physically rather
            // than over-mixing, so the equator-pole gradient stays Earth-like without heavy upper
            // drag (which a single slab had needed, at the cost of a static jet).
            wind_friction_upper: 8.0e-6,
            // Vertical momentum coupling (~19-hour timescale): below `wind_relax` so the jet's belt
            // shear survives, but strong enough to drag the turbulent surface variability aloft —
            // without it the upper layer (forced only by the slow upper temperature) is nearly steady.
            vertical_wind_coupling: 1.5e-5,
            // Storm-track eddy conductance — the dominant meridional heat transport (Budyko–Sellers
            // EBM closure for unresolved baroclinic-eddy flux). Strong enough that down-gradient
            // transport clearly overcomes the equatorward surface advection and warms the poles;
            // self-limiting because it scales with the (shrinking) gradient. Sized against the
            // diffusion CFL (see MAX_SUBSTEP). NOTE: this also *stabilizes* the surface advection's
            // central-difference form, so it cannot be lowered to offset the new upper-layer
            // transport — damp the upper eddies via `wind_friction_upper` instead.
            eddy_diffusivity: 150.0,

            // Per-layer greenhouse strength. Two stacked grey layers give a stronger total
            // greenhouse than one, so each emissivity is below the old single-slab 0.90; the upper
            // layer is optically thinner than the lower. Tuned (with the heat capacities + lapse) to
            // keep the surface mean near Earth's and a realistic lapse (lower ~255 K, upper ~225 K).
            atm_emissivity_lower: 0.72,
            atm_emissivity_upper: 0.50,
            // Column thermal mass split between the layers (sum ≈ the old 1e7 single slab); the lower
            // troposphere holds a bit more mass than the upper.
            atm_heat_capacity_lower: 6.0e6,
            atm_heat_capacity_upper: 4.0e6,
            // Sensible-heat exchange coefficient (W·m⁻²·K⁻¹) for each convective link of the chain.
            convection_coeff: 35.0,
            // Per-link lapse threshold (K): each layer settles ~33 K below the one beneath it, so the
            // chain builds surface → lower (−33) → upper (−66) — a realistic tropospheric profile —
            // while preserving the greenhouse.
            convection_threshold: 33.0,
            // The atmosphere mixes heat laterally more freely than the surface (both layers). This
            // also *stabilizes* the layers' advective central-difference form (the upper wind is fast
            // and divergent), so it must stay at this magnitude — cutting it to offset transport
            // sends the advection checkerboard-unstable.
            atm_diffusivity: 60.0,
            // Production: the upper layer is advected. (Ablation switch for diagnostics only.)
            upper_advection: true,
        }
    }
}

/// Lowest temperature a cell is allowed to reach (cosmic microwave background), keeping the
/// T⁴ term well-behaved through long nights.
const TEMP_FLOOR: f32 = 2.7;

/// Largest internal integration step (sim seconds). The engine sub-steps to this bound so the
/// explicit Euler update stays stable regardless of the wall-clock frame time. Chosen to keep the
/// diffusion update well within its stability limit dt·D·Σw/C < 1 — now sized for the *peak*
/// conductance (base `diffusivity` plus the storm-track `eddy_diffusivity` at the strongest
/// gradient), which lands the worst-case CFL factor around 0.7. See `diffusion_substep_is_stable`.
const MAX_SUBSTEP: f32 = 150.0;

/// Climate simulation state over a [`Grid`].
pub struct Sim {
    pub climate: Climate,
    /// Accumulated simulation time (seconds).
    pub time: f64,
    /// Tectonic plates and their motion. Empty until [`Sim::generate_terrain`] runs; geology and
    /// climate share this struct because they will couple (currents, orographic effects) later.
    pub terrain: Terrain,
    /// Crust + elevation field, derived from the plates. Empty until [`Sim::generate_terrain`].
    pub crust: Crust,
    /// Current per-cell surface temperatures (K).
    temp: Vec<f32>,
    /// Scratch buffer for the double-buffered surface update.
    scratch: Vec<f32>,
    /// Per-cell surface heat capacity (J·m⁻²·K⁻¹): the ocean mixed-layer value where the cell is
    /// below sea level, the land value above. Recomputed whenever the terrain changes; defaults to
    /// all-land before terrain is generated. Its ocean/land contrast is what damps the seasonal
    /// (and diurnal) swing realistically.
    heat_cap: Vec<f32>,
    /// Current per-cell **lower** atmosphere-layer temperatures (K).
    temp_l: Vec<f32>,
    /// Scratch buffer for the double-buffered lower-atmosphere update.
    temp_l_scratch: Vec<f32>,
    /// Current per-cell **upper** atmosphere-layer temperatures (K).
    temp_u: Vec<f32>,
    /// Scratch buffer for the double-buffered upper-atmosphere update.
    temp_u_scratch: Vec<f32>,
    /// Current per-cell surface wind (tangent to the sphere, m·s⁻¹). Prognostic: integrated from the
    /// momentum balance each step. Seeded from the diagnostic balance on the first [`Sim::advance`].
    /// Represents the lower-tropospheric flow — it advects both `T_s` and the lower layer `T_l`.
    wind: Vec<Vec3>,
    /// Scratch buffer for the double-buffered wind update.
    wind_scratch: Vec<Vec3>,
    /// Current per-cell **upper**-layer wind (tangent, m·s⁻¹): the subtropical jet + overturning
    /// return branch aloft. Prognostic like [`wind`]; advects the upper layer temperature `T_u`.
    wind_hi: Vec<Vec3>,
    /// Scratch buffer for the double-buffered upper-wind update.
    wind_hi_scratch: Vec<Vec3>,
    /// Current per-cell surface temperature gradient (tangent, K·m⁻¹), recomputed each step and
    /// shared by the surface wind's pressure-gradient force.
    grad: Vec<Vec3>,
    /// Current per-cell lower-atmosphere temperature gradient (tangent, K·m⁻¹); advects `T_l`.
    grad_l: Vec<Vec3>,
    /// Current per-cell upper-atmosphere temperature gradient (tangent, K·m⁻¹), driving the upper
    /// wind's pressure force and advecting `T_u`.
    grad_u: Vec<Vec3>,
    /// Current per-cell surface→lower convective (sensible-heat) flux H (W·m⁻²) — the upward venting
    /// off the surface, written each step. Diagnostic only (not fed back); the ITCZ/convection signal.
    conv: Vec<f32>,
    /// Whether the prognostic winds have been seeded from the diagnostic balance yet.
    wind_seeded: bool,
}

impl Sim {
    /// Create a simulation with all cells at a uniform starting temperature. The terrain starts
    /// empty; call [`Sim::generate_terrain`] (which needs the grid) to populate it.
    pub fn new(n: usize, climate: Climate, initial_temp: f32) -> Self {
        Self {
            climate,
            time: 0.0,
            terrain: Terrain::empty(),
            crust: Crust::empty(),
            temp: vec![initial_temp; n],
            scratch: vec![initial_temp; n],
            // No terrain yet → treat the whole surface as land; recomputed in `generate_terrain`.
            heat_cap: vec![climate.heat_capacity; n],
            // Each atmosphere layer starts cooler than the one below (it sits above it): the lower
            // layer ~Γ below the surface, the upper ~2Γ, the chain's equilibrium ordering.
            temp_l: vec![initial_temp - 33.0; n],
            temp_l_scratch: vec![initial_temp - 33.0; n],
            temp_u: vec![initial_temp - 66.0; n],
            temp_u_scratch: vec![initial_temp - 66.0; n],
            wind: vec![Vec3::ZERO; n],
            wind_scratch: vec![Vec3::ZERO; n],
            wind_hi: vec![Vec3::ZERO; n],
            wind_hi_scratch: vec![Vec3::ZERO; n],
            grad: vec![Vec3::ZERO; n],
            grad_l: vec![Vec3::ZERO; n],
            grad_u: vec![Vec3::ZERO; n],
            conv: vec![0.0; n],
            wind_seeded: false,
        }
    }

    /// Generate the tectonic plates and the elevation field they imply over `grid` (deterministic
    /// in `seed`). The crust is derived from the plates, so it is generated in the same pass.
    pub fn generate_terrain(&mut self, grid: &Grid, num_plates: usize, seed: u64) {
        self.terrain = Terrain::generate(grid, num_plates, seed);
        self.crust = Crust::generate(grid, &self.terrain, seed);
        self.recompute_heat_capacity();
    }

    /// Set each cell's surface heat capacity from its elevation: the ocean mixed-layer value below
    /// sea level, the land value above. Called whenever the terrain (hence the land/sea mask)
    /// changes. Ocean cells barely change temperature through the day or season; this is what keeps
    /// the hemispheres from swinging ±50 °C with the season.
    fn recompute_heat_capacity(&mut self) {
        let (land, ocean) = (self.climate.heat_capacity, self.climate.heat_capacity_ocean);
        for (hc, &elev) in self.heat_cap.iter_mut().zip(self.crust.elevation.iter()) {
            *hc = if elev < 0.0 { ocean } else { land };
        }
    }

    /// Current per-cell surface temperatures (K).
    pub fn temperatures(&self) -> &[f32] {
        &self.temp
    }

    /// Current per-cell lower-atmosphere-layer temperatures (K).
    pub fn lower_atmosphere_temperatures(&self) -> &[f32] {
        &self.temp_l
    }

    /// Current per-cell upper-atmosphere-layer temperatures (K).
    pub fn upper_atmosphere_temperatures(&self) -> &[f32] {
        &self.temp_u
    }

    /// Per-cell surface elevation (m relative to sea level; negative = ocean floor).
    pub fn elevations(&self) -> &[f32] {
        &self.crust.elevation
    }

    /// Current per-cell surface wind (tangent to the sphere, m·s⁻¹). All zero until the first
    /// [`Sim::advance`] has run.
    pub fn winds(&self) -> &[Vec3] {
        &self.wind
    }

    /// Current per-cell upper-layer wind (tangent, m·s⁻¹): the subtropical jet + overturning return
    /// branch aloft. All zero until the first [`Sim::advance`] has run.
    pub fn winds_hi(&self) -> &[Vec3] {
        &self.wind_hi
    }

    /// Current per-cell convective heat flux H (W·m⁻²) — the upward venting into the atmosphere.
    /// Zero until the first [`Sim::advance`]; highlights the convergence/convection zones (ITCZ).
    pub fn convection(&self) -> &[f32] {
        &self.conv
    }

    /// Unit direction toward the sun in the planet's frame, at the given sim time.
    pub fn sun_direction(&self, time: f64) -> Vec3 {
        let c = &self.climate;
        // Subsolar longitude sweeps once per day (planet spins eastward → sun drifts west).
        let lon = -std::f32::consts::TAU * (time / c.day_seconds as f64) as f32;
        // Solar declination oscillates once per year between ±obliquity.
        let decl = c.obliquity * (std::f32::consts::TAU * (time / c.year_seconds as f64) as f32).sin();
        let (s_decl, c_decl) = decl.sin_cos();
        let (s_lon, c_lon) = lon.sin_cos();
        Vec3::new(c_decl * c_lon, c_decl * s_lon, s_decl)
    }

    /// Advance the simulation by `sim_seconds`, sub-stepping for numerical stability.
    pub fn advance(&mut self, grid: &Grid, sim_seconds: f32) {
        if sim_seconds <= 0.0 {
            return;
        }
        let mut remaining = sim_seconds;
        while remaining > 0.0 {
            let dt = remaining.min(MAX_SUBSTEP);
            let sun = self.sun_direction(self.time);

            // The single equator-to-pole gradient factor scales both the belts and the storm-track
            // eddy diffusivity; compute it (and the temperature gradient that drives the wind's
            // pressure force) once and share.
            let gradient = wind::gradient_factor(grid, &self.temp);
            // Three gradients: ∇T_s drives the surface wind (and advects T_s), ∇T_l advects the
            // lower layer (carried by the surface wind), ∇T_u drives the upper wind (and advects T_u).
            let radius = self.climate.planet_radius;
            self.grad = wind::gradients(grid, &self.temp, radius);
            self.grad_l = wind::gradients(grid, &self.temp_l, radius);
            self.grad_u = wind::gradients(grid, &self.temp_u, radius);

            // Seed both prognostic winds from the diagnostic balance the first time, so they start
            // in a sensible state rather than from rest.
            if !self.wind_seeded {
                wind::compute(grid, &self.climate, gradient, &self.grad, &mut self.wind, wind::WindLayer::Surface);
                wind::compute(grid, &self.climate, gradient, &self.grad_u, &mut self.wind_hi, wind::WindLayer::Upper);
                self.wind_seeded = true;
            }

            // Integrate both wind layers one step (momentum balance: advection + Coriolis + pressure
            // + belt relaxation + viscosity), then swap them in. The surface relaxes to the tri-cell
            // belts and is driven by ∇T_s; the upper layer relaxes to the jet/return target and is
            // driven by the upper-layer ∇T_u.
            wind::step(
                grid,
                &self.climate,
                gradient,
                &self.grad,
                dt,
                &self.wind,
                &mut self.wind_scratch,
                wind::WindLayer::Surface,
                None,
            );
            std::mem::swap(&mut self.wind, &mut self.wind_scratch);
            // The upper wind is vertically coupled to the (now-updated) surface wind, inheriting its
            // turbulence so the jet is lively.
            wind::step(
                grid,
                &self.climate,
                gradient,
                &self.grad_u,
                dt,
                &self.wind_hi,
                &mut self.wind_hi_scratch,
                wind::WindLayer::Upper,
                Some(&self.wind),
            );
            std::mem::swap(&mut self.wind_hi, &mut self.wind_hi_scratch);

            // Area-mean of the upper-layer advective tendency −u_hi·∇T_u. The upper wind is fast and
            // divergent, so this mean is nonzero and would otherwise drift the global upper-layer
            // temperature (and with it the greenhouse); subtract it in the kernel so advection only
            // redistributes heat — the single-layer stand-in for the vertical mass compensation that
            // makes the continuum horizontal advection conserve energy. (The lower layer rides the
            // slower surface wind, so like `T_s` it tolerates the small uncorrected leak.)
            let (mut adv_num, mut adv_den) = (0.0f64, 0.0f64);
            for i in 0..grid.n {
                let a = grid.areas[i] as f64;
                adv_num += (-self.wind_hi[i].dot(self.grad_u[i])) as f64 * a;
                adv_den += a;
            }
            let adv_u_correction = (adv_num / adv_den) as f32;

            temperature::step(
                grid,
                &self.climate,
                sun,
                dt,
                TEMP_FLOOR,
                gradient,
                temperature::Temps {
                    s: &self.temp,
                    l: &self.temp_l,
                    u: &self.temp_u,
                },
                &self.heat_cap,
                &self.wind,
                &self.grad_l,
                &self.wind_hi,
                &self.grad_u,
                adv_u_correction,
                temperature::Outs {
                    s: &mut self.scratch,
                    l: &mut self.temp_l_scratch,
                    u: &mut self.temp_u_scratch,
                    conv: &mut self.conv,
                },
            );
            std::mem::swap(&mut self.temp, &mut self.scratch);
            std::mem::swap(&mut self.temp_l, &mut self.temp_l_scratch);
            std::mem::swap(&mut self.temp_u, &mut self.temp_u_scratch);
            self.time += dt as f64;
            remaining -= dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_direction_is_unit_and_sweeps() {
        let sim = Sim::new(10, Climate::default(), 255.0);
        let s0 = sim.sun_direction(0.0);
        assert!((s0.length() - 1.0).abs() < 1e-5);
        // After a quarter day the subsolar longitude has rotated ~90°.
        let s_quarter = sim.sun_direction((sim.climate.day_seconds / 4.0) as f64);
        assert!(s0.dot(s_quarter).abs() < 0.1, "quarter-day sun should be ~orthogonal");
        // After a full day it returns close to the start.
        let s_full = sim.sun_direction(sim.climate.day_seconds as f64);
        assert!(s0.dot(s_full) > 0.999, "full-day sun should return to start");
    }

    #[test]
    fn subsolar_cell_approaches_radiative_equilibrium() {
        // A tiny grid, sun fixed (no daily motion) by making a day extremely long, and no
        // diffusion: the most-lit cell should converge toward its radiative equilibrium.
        let g = Grid::new(600);
        let mut climate = Climate::default();
        climate.diffusivity = 0.0;
        // Disable all lateral transport so the most-lit cell reaches *pure* radiative equilibrium.
        climate.eddy_diffusivity = 0.0;
        climate.trade_wind_speed = 0.0;
        climate.meridional_wind_speed = 0.0;
        climate.geostrophic_coeff = 0.0;
        // Disable the atmosphere too (both grey layers + convection) so the surface reaches the
        // bare σT⁴ = absorbed equilibrium this test checks.
        climate.atm_emissivity_lower = 0.0;
        climate.atm_emissivity_upper = 0.0;
        climate.convection_coeff = 0.0;
        climate.day_seconds = 1.0e12; // effectively frozen sun
        climate.year_seconds = 1.0e15;
        let mut sim = Sim::new(g.n, climate, 200.0);
        // Run for a long time to settle.
        for _ in 0..4000 {
            sim.advance(&g, MAX_SUBSTEP);
        }
        let sun = sim.sun_direction(0.0);
        // Find the most-lit cell and its expected equilibrium.
        let (best, _) = g
            .centers
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.dot(sun)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let cos = g.centers[best].dot(sun).max(0.0);
        let absorbed = (1.0 - climate.albedo) * climate.solar_constant * cos;
        let expected = (absorbed / (climate.emissivity * STEFAN_BOLTZMANN)).powf(0.25);
        let got = sim.temperatures()[best];
        assert!(
            (got - expected).abs() / expected < 0.02,
            "subsolar cell {got}K vs equilibrium {expected}K",
        );
    }

    #[test]
    fn diffusion_substep_is_stable() {
        // Explicit-Euler diffusion is stable when dt·D·max(Σⱼwᵢⱼ)/C < 1 (well under 2). The peak
        // conductance is the base diffusivity plus the storm-track eddy term at its strongest:
        // `eddy_diffusivity` × the maximum gradient factor (clamped to 2.0) × the storm-band peak
        // (1.0). Sizing the substep against this worst case keeps every cell stable.
        let g = Grid::new(8000);
        let c = Climate::default();
        let max_wsum = (0..g.n)
            .map(|i| g.neighbor_weights(i).iter().sum::<f32>())
            .fold(0.0f32, f32::max);
        let max_d = c.diffusivity + c.eddy_diffusivity * 2.0;
        let factor = MAX_SUBSTEP * max_d * max_wsum / c.heat_capacity;
        assert!(factor < 1.0, "diffusion CFL factor {factor} (max Σw = {max_wsum}, D = {max_d}) is unstable");
    }

    #[test]
    fn temperatures_stay_finite_and_bounded_over_a_day() {
        let g = Grid::new(1500);
        let mut sim = Sim::new(g.n, Climate::default(), 255.0);
        // Simulate two full days.
        let day = sim.climate.day_seconds;
        for _ in 0..200 {
            sim.advance(&g, day * 2.0 / 200.0);
        }
        for &t in sim.temperatures() {
            assert!(t.is_finite(), "temperature went non-finite");
            assert!((TEMP_FLOOR..600.0).contains(&t), "temperature out of range: {t}");
        }
    }
}
