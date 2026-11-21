//! Vulkan/DirectX12 GPU compute backend via `wgpu` for matrix multiplication (`gemm`).

use crate::tensor::{Backend, CPU_BACKEND};
use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

const GEMM_WGSL: &str = r#"
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        acc = acc + a[row * dims.k + i] * b[i * dims.n + col];
    }
    c[row * dims.n + col] = c[row * dims.n + col] + acc;
}
"#;

struct BufferCache {
    dims: wgpu::Buffer,
    a: wgpu::Buffer,
    a_cap: u64,
    b: wgpu::Buffer,
    b_cap: u64,
    c: wgpu::Buffer,
    read: wgpu::Buffer,
    c_cap: u64,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    adapter_name: String,
    cache: Mutex<BufferCache>,
}

static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

fn make_buf(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn init_gpu() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let adapter_name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("neuro-vulkan"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("neuro-gemm"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_WGSL)),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("neuro-gemm-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("neuro-gemm-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("neuro-gemm-pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let init_cap = 64 * 1024;
    let cache = BufferCache {
        dims: make_buf(&device, 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
        a: make_buf(&device, init_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
        a_cap: init_cap,
        b: make_buf(&device, init_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
        b_cap: init_cap,
        c: make_buf(
            &device,
            init_cap,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        ),
        read: make_buf(&device, init_cap, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        c_cap: init_cap,
    };

    Some(GpuContext {
        device,
        queue,
        pipeline,
        bgl,
        adapter_name,
        cache: Mutex::new(cache),
    })
}

fn ctx() -> Option<&'static GpuContext> {
    GPU_CTX.get_or_init(init_gpu).as_ref()
}

pub fn gpu_device_name() -> Option<&'static str> {
    ctx().map(|c| c.adapter_name.as_str())
}

fn f32_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn u32_bytes(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn try_vulkan_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<(), ()> {
    let g = ctx().ok_or(())?;
    let a_len = m.checked_mul(k).ok_or(())?;
    let b_len = k.checked_mul(n).ok_or(())?;
    let c_len = m.checked_mul(n).ok_or(())?;
    if a.len() < a_len || b.len() < b_len || c.len() < c_len {
        return Err(());
    }

    let a_size = (a_len * 4) as u64;
    let b_size = (b_len * 4) as u64;
    let c_size = (c_len * 4) as u64;

    // Grow cached buffers on demand instead of reallocating on every GEMM dispatch.
    let mut cache = g.cache.lock().map_err(|_| ())?;
    if a_size > cache.a_cap {
        let cap = a_size.next_power_of_two();
        cache.a = make_buf(&g.device, cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        cache.a_cap = cap;
    }
    if b_size > cache.b_cap {
        let cap = b_size.next_power_of_two();
        cache.b = make_buf(&g.device, cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        cache.b_cap = cap;
    }
    if c_size > cache.c_cap {
        let cap = c_size.next_power_of_two();
        cache.c = make_buf(
            &g.device,
            cap,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        cache.read = make_buf(&g.device, cap, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        cache.c_cap = cap;
    }

    g.queue.write_buffer(&cache.dims, 0, u32_bytes(&[m as u32, k as u32, n as u32, 0]));
    g.queue.write_buffer(&cache.a, 0, f32_bytes(&a[..a_len]));
    g.queue.write_buffer(&cache.b, 0, f32_bytes(&b[..b_len]));
    g.queue.write_buffer(&cache.c, 0, f32_bytes(&c[..c_len]));

    let bind_group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cache.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cache.a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cache.b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cache.c.as_entire_binding() },
        ],
    });

    let mut encoder = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wx = (n as u32 + 15) / 16;
        let wy = (m as u32 + 15) / 16;
        pass.dispatch_workgroups(wx, wy, 1);
    }
    encoder.copy_buffer_to_buffer(&cache.c, 0, &cache.read, 0, c_size);
    g.queue.submit(Some(encoder.finish()));

    let slice = cache.read.slice(0..c_size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    g.device.poll(wgpu::Maintain::Wait);
    rx.recv().map_err(|_| ())?.map_err(|_| ())?;

    let view = slice.get_mapped_range();
    for (dst, chunk) in c[..c_len].iter_mut().zip(view.chunks_exact(4)) {
        *dst = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    drop(view);
    cache.read.unmap();
    Ok(())
}

pub struct GpuBackend;
pub static GPU_BACKEND: GpuBackend = GpuBackend;

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        "vulkan-gpu"
    }

    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, simd: bool) {
        if m == 0 || k == 0 || n == 0 {
            return;
        }
        // Below 32k operations, CPU execution avoids PCIe transfer overhead.
        if m * k * n < 32_768 || try_vulkan_gemm(a, b, c, m, k, n).is_err() {
            CPU_BACKEND.gemm(a, b, c, m, k, n, simd);
        }
    }
}