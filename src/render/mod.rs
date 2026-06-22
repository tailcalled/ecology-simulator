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
/// data, keyed on `index()`: continuous layers (e.g. `Temperature`) go through a colormap over
/// `range()`, while `Plates` is categorical and gets a per-id palette color instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Temperature,
    Plates,
}

impl Layer {
    /// Parse the layer id sent from the UI radio buttons.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temperature" => Some(Layer::Temperature),
            "plates" => Some(Layer::Plates),
            _ => None,
        }
    }

    /// Layer index passed to the shader (selects colormap vs. plate palette).
    pub fn index(self) -> u32 {
        match self {
            Layer::Temperature => 0,
            Layer::Plates => 1,
        }
    }

    /// Default (min, max) data range for the colormap, in the layer's units. Unused for
    /// categorical layers like `Plates`.
    pub fn range(self) -> (f32, f32) {
        match self {
            // Kelvin: ~ −63 °C (210 K) to ~ 77 °C (350 K).
            Layer::Temperature => (210.0, 350.0),
            Layer::Plates => (0.0, 1.0),
        }
    }
}
