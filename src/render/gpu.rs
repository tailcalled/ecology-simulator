//! WebGPU renderer (wasm-only). Owns a single `wgpu` device shared across the two views; each
//! view is a surface over an `OffscreenCanvas` with its own camera/layer uniform and depth
//! buffer. Geometry and the per-cell data buffer are shared between views — only the per-view
//! camera uniform differs. All GPU objects stay on this one worker thread (they are
//! `!Send`/`!Sync` under wasm atomics).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use wasm_bindgen::JsValue;
use wgpu::util::DeviceExt;
use web_sys::OffscreenCanvas;

use super::mesh::Vertex;
use super::{Layer, Projection};

/// Per-view camera/overlay uniform (std140: mat4 + 4·vec4 = 128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    eye: [f32; 4],
    sun: [f32; 4],
    params: [f32; 4],    // x = data min, y = data max, z = show sunlight, w = show graticule
    highlight: [f32; 4], // x = hovered cell index, or -1 for none
}

/// What the engine supplies per view each frame.
#[derive(Clone, Copy)]
pub struct ViewCamera {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// How the vertex shader maps the sphere to the screen for this view.
    pub projection: Projection,
}

struct View {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    camera_buf: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    layer: Layer,
    show_sunlight: bool,
    show_graticule: bool,
    show_motion: bool,
    /// Cell highlighted by the cursor hovering this view, if any.
    highlight: Option<u32>,
}

pub struct Renderer {
    // Must outlive the surfaces: on the web, dropping the Instance invalidates surfaces
    // created from it ("A valid external Instance reference no longer exists").
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    cell_buf: wgpu::Buffer,
    cell_bind_group: wgpu::BindGroup,
    /// Per-cell plate id (static after terrain generation); colored via the plate palette.
    plate_buf: wgpu::Buffer,
    plate_bind_group: wgpu::BindGroup,
    /// Per-cell center (lon, lat) in radians (static); read by the map-projection vertex shader.
    center_buf: wgpu::Buffer,
    center_bind_group: wgpu::BindGroup,
    marker_pipeline: wgpu::RenderPipeline,
    marker_buf: wgpu::Buffer,
    marker_capacity: u32,
    marker_count: u32,
    /// Plate-motion arrow field (line list), shared across views, toggled per view.
    arrow_pipeline: wgpu::RenderPipeline,
    arrow_buf: wgpu::Buffer,
    arrow_capacity: u32,
    arrow_count: u32,
    views: Vec<View>,
}

/// Max vertices in the zoom-footprint outline line strip.
const MARKER_CAPACITY: u32 = 64;
/// Max vertices in the plate-motion arrow line list (6 per arrow → ~680 arrows).
const ARROW_CAPACITY: u32 = 4096;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Background (space) clear color.
const CLEAR: wgpu::Color = wgpu::Color { r: 0.01, g: 0.012, b: 0.02, a: 1.0 };

/// Whether `navigator.gpu` is exposed in the current (worker) global scope.
fn webgpu_available() -> bool {
    let global = js_sys::global();
    js_sys::Reflect::get(&global, &JsValue::from_str("navigator"))
        .ok()
        .and_then(|nav| js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")).ok())
        .map(|g| !g.is_undefined() && !g.is_null())
        .unwrap_or(false)
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Renderer {
    pub async fn new(
        canvas0: OffscreenCanvas,
        canvas1: OffscreenCanvas,
        mesh: &[Vertex],
        n_cells: usize,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // Is WebGPU even exposed in this worker?
        log::info!("renderer: worker navigator.gpu present = {}", webgpu_available());

        // Acquire an adapter. `compatible_surface` is ignored by the WebGPU backend (it just
        // calls navigator.gpu.requestAdapter), so we try a few power-preference / fallback
        // variants and log each, rather than failing on the first null result.
        let attempts = [
            (wgpu::PowerPreference::HighPerformance, false, "high-performance"),
            (wgpu::PowerPreference::None, false, "default"),
            (wgpu::PowerPreference::LowPower, false, "low-power"),
            (wgpu::PowerPreference::LowPower, true, "software-fallback"),
        ];
        let mut adapter = None;
        for (pref, fallback, label) in attempts {
            match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: pref,
                    force_fallback_adapter: fallback,
                    compatible_surface: None,
                })
                .await
            {
                Ok(a) => {
                    log::info!("renderer: adapter acquired via '{label}'");
                    adapter = Some(a);
                    break;
                }
                Err(e) => log::warn!("renderer: adapter attempt '{label}' failed: {e}"),
            }
        }
        let adapter = adapter.ok_or_else(|| {
            "no WebGPU adapter (navigator.gpu.requestAdapter returned null for every variant). \
             If WebGPU worked earlier this session, the GPU process is likely wedged — fully \
             quit the browser (ensure no background processes survive) and reopen, then load \
             the page once."
                .to_string()
        })?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ecology-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;

        let sizes = [
            (canvas0.width().max(1), canvas0.height().max(1)),
            (canvas1.width().max(1), canvas1.height().max(1)),
        ];
        let surfaces = [
            instance
                .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas0))
                .map_err(|e| format!("surface0: {e}"))?,
            instance
                .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas1))
                .map_err(|e| format!("surface1: {e}"))?,
        ];

        // Use the surface's preferred format. On the web a WebGPU canvas can only be
        // configured with its preferred (non-sRGB) format — configuring with an `*-srgb`
        // format is invalid and the surface silently never presents. (sRGB output, if wanted,
        // must go through `view_formats` + an sRGB texture view instead.)
        let caps = surfaces[0].get_capabilities(&adapter);
        let format = caps.formats[0];
        log::info!("renderer: surface format = {format:?}, {} cells", n_cells);

        // --- Bind group layouts ---
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let storage_bgl_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cell_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cell-bgl"),
            entries: &[storage_bgl_entry],
        });
        let plate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plate-bgl"),
            entries: &[storage_bgl_entry],
        });
        let center_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("center-bgl"),
            entries: &[storage_bgl_entry],
        });

        // --- Shared geometry + per-cell data ---
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell-mesh"),
            contents: bytemuck::cast_slice(mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cell_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell-data"),
            size: (n_cells.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cell-bg"),
            layout: &cell_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cell_buf.as_entire_binding(),
            }],
        });
        // Per-cell plate id (one u32 each). Uploaded once after terrain generation; zero-filled
        // until then so the buffer is always bound and the pipeline is valid.
        let plate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plate-data"),
            size: (n_cells.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plate-bg"),
            layout: &plate_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: plate_buf.as_entire_binding(),
            }],
        });
        // Per-cell center (lon, lat) radians as a vec2<f32> each. Uploaded once after grid build;
        // zero-filled until then so the buffer is always bound and the pipeline is valid.
        let center_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("center-data"),
            size: (n_cells.max(1) * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let center_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("center-bg"),
            layout: &center_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: center_buf.as_entire_binding(),
            }],
        });

        // --- Pipeline ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cells-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cells.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[
                Some(&camera_bgl),
                Some(&cell_bgl),
                Some(&plate_bgl),
                Some(&center_bgl),
            ],
            immediate_size: 0,
        });
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cells-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // opaque convex sphere; depth test resolves overdraw
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- Marker pipeline (zoom-footprint outline; line strip, camera uniform only) ---
        let marker_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marker-layout"),
            bind_group_layouts: &[Some(&camera_bgl)],
            immediate_size: 0,
        });
        let marker_attrs = wgpu::vertex_attr_array![0 => Float32x3];
        let marker_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("marker-pipeline"),
            layout: Some(&marker_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_marker"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (3 * std::mem::size_of::<f32>()) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &marker_attrs,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false), // don't occlude itself; lift handles ordering
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_marker"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let marker_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("marker-verts"),
            size: (MARKER_CAPACITY as usize * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Arrow pipeline (plate-motion field; line list, same pos-only vertex + camera-only
        // bind group as the marker, but its own fragment color and topology). ---
        let arrow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("arrow-pipeline"),
            layout: Some(&marker_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_marker"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (3 * std::mem::size_of::<f32>()) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &marker_attrs,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Depth-tested (the sphere occludes far-side arrows) but doesn't write depth.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_arrow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let arrow_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arrow-verts"),
            size: (ARROW_CAPACITY as usize * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Per-view resources ---
        let default_layers = [Layer::Temperature, Layer::Temperature];
        let mut views = Vec::with_capacity(2);
        for (i, surface) in surfaces.into_iter().enumerate() {
            let (w, h) = sizes[i];
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
            };
            surface.configure(&device, &config);
            let depth = make_depth(&device, w, h);
            let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera-uniform"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera-bg"),
                layout: &camera_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                }],
            });
            log::info!("renderer: view {i} surface configured at {w}x{h}");
            views.push(View {
                surface,
                config,
                depth,
                camera_buf,
                camera_bind_group,
                layer: default_layers[i],
                show_sunlight: true,
                show_graticule: true,
                show_motion: false,
                highlight: None,
            });
        }

        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipeline,
            vertex_buf,
            vertex_count: mesh.len() as u32,
            cell_buf,
            cell_bind_group,
            plate_buf,
            plate_bind_group,
            center_buf,
            center_bind_group,
            marker_pipeline,
            marker_buf,
            marker_capacity: MARKER_CAPACITY,
            marker_count: 0,
            arrow_pipeline,
            arrow_buf,
            arrow_capacity: ARROW_CAPACITY,
            arrow_count: 0,
            views,
        })
    }

    pub fn resize(&mut self, view: usize, width: u32, height: u32) {
        if let Some(v) = self.views.get_mut(view) {
            v.config.width = width.max(1);
            v.config.height = height.max(1);
            v.surface.configure(&self.device, &v.config);
            v.depth = make_depth(&self.device, v.config.width, v.config.height);
        }
    }

    pub fn set_layer(&mut self, view: usize, layer: Layer) {
        if let Some(v) = self.views.get_mut(view) {
            v.layer = layer;
        }
    }

    /// Toggle an overlay ("sunlight" or "graticule") for a view.
    pub fn set_overlay(&mut self, view: usize, which: &str, enabled: bool) {
        if let Some(v) = self.views.get_mut(view) {
            match which {
                "sunlight" => v.show_sunlight = enabled,
                "graticule" => v.show_graticule = enabled,
                "motion" => v.show_motion = enabled,
                _ => {}
            }
        }
    }

    /// Set (or clear) the cell highlighted by the cursor hovering a view.
    pub fn set_highlight(&mut self, view: usize, cell: Option<u32>) {
        if let Some(v) = self.views.get_mut(view) {
            v.highlight = cell;
        }
    }

    /// Upload the zoom-footprint outline (world positions) drawn on the globe view.
    pub fn set_marker(&mut self, points: &[Vec3]) {
        let n = points.len().min(self.marker_capacity as usize);
        let flat: Vec<f32> = points[..n].iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        self.queue
            .write_buffer(&self.marker_buf, 0, bytemuck::cast_slice(&flat));
        self.marker_count = n as u32;
    }

    /// Upload the per-cell plate ids (static after terrain generation), read by the plate layer.
    pub fn upload_plate_data(&self, ids: &[u32]) {
        self.queue
            .write_buffer(&self.plate_buf, 0, bytemuck::cast_slice(ids));
    }

    /// Upload the per-cell center (lon, lat) radians (static after grid build), read by the
    /// map-projection vertex shader to unwrap cells across the antimeridian.
    pub fn upload_cell_centers(&self, centers: &[Vec2]) {
        self.queue
            .write_buffer(&self.center_buf, 0, bytemuck::cast_slice(centers));
    }

    /// Upload the plate-motion arrow field (line-list world positions). Static after generation;
    /// drawn per view when the motion overlay is on.
    pub fn set_arrows(&mut self, points: &[Vec3]) {
        let n = points.len().min(self.arrow_capacity as usize);
        let flat: Vec<f32> = points[..n].iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        self.queue
            .write_buffer(&self.arrow_buf, 0, bytemuck::cast_slice(&flat));
        self.arrow_count = n as u32;
    }

    pub fn aspect(&self, view: usize) -> f32 {
        self.views
            .get(view)
            .map(|v| v.config.width as f32 / v.config.height.max(1) as f32)
            .unwrap_or(1.0)
    }

    /// Upload the latest per-cell data (e.g. temperatures) to the shared storage buffer.
    pub fn upload_cell_data(&self, data: &[f32]) {
        self.queue
            .write_buffer(&self.cell_buf, 0, bytemuck::cast_slice(data));
    }

    /// Render both views with the supplied per-view cameras and current sun direction.
    pub fn render(&mut self, cameras: [ViewCamera; 2], sun: Vec3) {
        for (i, v) in self.views.iter().enumerate() {
            let (lo, hi) = v.layer.range();
            let cam = &cameras[i];
            let uniform = CameraUniform {
                view_proj: cam.view_proj.to_cols_array(),
                eye: [cam.eye.x, cam.eye.y, cam.eye.z, 1.0],
                sun: [sun.x, sun.y, sun.z, 0.0],
                params: [
                    lo,
                    hi,
                    if v.show_sunlight { 1.0 } else { 0.0 },
                    if v.show_graticule { 1.0 } else { 0.0 },
                ],
                highlight: [
                    v.highlight.map_or(-1.0, |c| c as f32),
                    v.layer.index() as f32,
                    cam.projection.index() as f32,
                    0.0,
                ],
            };
            self.queue
                .write_buffer(&v.camera_buf, 0, bytemuck::bytes_of(&uniform));

            use wgpu::CurrentSurfaceTexture as Cst;
            let frame = match v.surface.get_current_texture() {
                Cst::Success(f) | Cst::Suboptimal(f) => f,
                Cst::Timeout => continue,
                _ => {
                    log::warn!("view {i}: surface unavailable, reconfiguring");
                    v.surface.configure(&self.device, &v.config);
                    continue;
                }
            };
            let target = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("cells"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(CLEAR),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &v.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &v.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.cell_bind_group, &[]);
                pass.set_bind_group(2, &self.plate_bind_group, &[]);
                pass.set_bind_group(3, &self.center_bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                pass.draw(0..self.vertex_count, 0..1);

                // Plate-motion arrows, drawn on any view with the motion overlay enabled.
                if v.show_motion && self.arrow_count > 1 {
                    pass.set_pipeline(&self.arrow_pipeline);
                    pass.set_bind_group(0, &v.camera_bind_group, &[]);
                    pass.set_vertex_buffer(0, self.arrow_buf.slice(..));
                    pass.draw(0..self.arrow_count, 0..1);
                }

                // The zoom-footprint outline is drawn only on the globe view (index 0).
                if i == 0 && self.marker_count > 1 {
                    pass.set_pipeline(&self.marker_pipeline);
                    pass.set_bind_group(0, &v.camera_bind_group, &[]);
                    pass.set_vertex_buffer(0, self.marker_buf.slice(..));
                    pass.draw(0..self.marker_count, 0..1);
                }
            }
            self.queue.submit(Some(encoder.finish()));
            frame.present();
        }
    }
}
