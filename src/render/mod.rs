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

/// A selectable visibility layer. Each view picks one; the shader colors cells by the layer's
/// data through a colormap. Only `Temperature` exists now — adding a layer means a new
/// per-cell data source + a colormap branch keyed on `index()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Temperature,
}

impl Layer {
    /// Parse the layer id sent from the UI `<select>`.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temperature" => Some(Layer::Temperature),
            _ => None,
        }
    }

    /// Colormap index passed to the shader.
    pub fn index(self) -> u32 {
        match self {
            Layer::Temperature => 0,
        }
    }

    /// Default (min, max) data range for the colormap, in the layer's units.
    pub fn range(self) -> (f32, f32) {
        match self {
            // Kelvin: ~ −63 °C (210 K) to ~ 77 °C (350 K).
            Layer::Temperature => (210.0, 350.0),
        }
    }
}
