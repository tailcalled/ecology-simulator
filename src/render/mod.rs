//! Rendering. The cross-platform geometry/camera math (`mesh`, `camera`) is testable on the
//! host; the actual WebGPU device + pipelines (`gpu`) are wasm-only because `wgpu`'s web
//! objects are `!Send`/`!Sync` once wasm atomics are enabled and must stay on one worker.

pub mod camera;
pub mod mesh;
pub mod pick;

#[cfg(target_arch = "wasm32")]
mod gpu;
#[cfg(target_arch = "wasm32")]
pub use gpu::{Renderer, ViewCamera};

/// How a view maps the unit sphere to the screen. `Sphere` is the orbiting 3D globe; the map
/// projections flatten the planet to a static 2D map in the vertex shader (see `cells.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Projection {
    /// 3D globe rendered with the view's perspective camera.
    #[default]
    Sphere,
    /// Winkel Tripel: a compromise map projection (low distortion in area, direction, distance).
    WinkelTripel,
}

impl Projection {
    /// Parse the projection id sent from the UI.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "sphere" => Some(Projection::Sphere),
            "winkel" => Some(Projection::WinkelTripel),
            _ => None,
        }
    }

    /// Mode index passed to the shader (selects the vertex-stage projection).
    pub fn index(self) -> u32 {
        match self {
            Projection::Sphere => 0,
            Projection::WinkelTripel => 1,
        }
    }
}

/// A selectable visibility layer. Each view picks one; the shader colors cells by the layer's
/// data, keyed on `index()`: continuous layers (e.g. `Temperature`) go through a colormap over
/// `range()`, while `Plates` is categorical and gets a per-id palette color instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Temperature,
    Plates,
    /// Surface elevation through a hypsometric (bathymetry/topography) colormap.
    Elevation,
}

impl Layer {
    /// Parse the layer id sent from the UI radio buttons.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temperature" => Some(Layer::Temperature),
            "plates" => Some(Layer::Plates),
            "elevation" => Some(Layer::Elevation),
            _ => None,
        }
    }

    /// Layer index passed to the shader (selects which per-cell field + colormap to use).
    pub fn index(self) -> u32 {
        match self {
            Layer::Temperature => 0,
            Layer::Plates => 1,
            Layer::Elevation => 2,
        }
    }

    /// Default (min, max) data range for the colormap, in the layer's units. Unused for
    /// categorical layers like `Plates`.
    pub fn range(self) -> (f32, f32) {
        match self {
            // Kelvin: ~ −63 °C (210 K) to ~ 77 °C (350 K).
            Layer::Temperature => (210.0, 350.0),
            Layer::Plates => (0.0, 1.0),
            // Metres: deepest trench to highest peak; the colormap pivots at 0 (sea level).
            Layer::Elevation => (-8000.0, 6000.0),
        }
    }
}

#[cfg(test)]
mod tests {
    /// Parse and validate the cell shader on the host. wgpu only compiles WGSL at runtime in the
    /// browser, so without this a shader typo (a missing binding, a type error, an undeclared
    /// varying) would only surface as a blank canvas in a GPU-backed browser session. naga is the
    /// very validator wgpu uses internally.
    #[test]
    fn cells_shader_is_valid_wgsl() {
        let src = include_str!("shaders/cells.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parse error in cells.wgsl");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("WGSL validation error in cells.wgsl");
    }
}
