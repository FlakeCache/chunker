use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy)]
pub enum ChunkingBackend {
    /// Always use the CPU FastCDC implementation.
    Cpu,
    /// Prefer GPU offload when compiled with `--features gpu` and a runtime device is available.
    Auto,
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum ChunkingError {
    #[error(
        "bounds_check_failed: offset {offset} + length {length} exceeds data length {data_len}"
    )]
    Bounds {
        data_len: usize,
        offset: usize,
        length: usize,
    },
    #[error("gpu_backend_unavailable: {reason}")]
    GpuUnavailable { reason: &'static str },
}

/// Validate slice bounds to prevent out-of-bounds access
/// Returns an error if offset + length would exceed `data_len` or overflow
fn validate_slice_bounds(
    data_len: usize,
    offset: usize,
    length: usize,
) -> Result<(), ChunkingError> {
    if offset.checked_add(length).is_none_or(|end| end > data_len) {
        return Err(ChunkingError::Bounds {
            data_len,
            offset,
            length,
        });
    }
    Ok(())
}

pub trait RollingHashBackend {
    fn name(&self) -> &'static str;

    fn is_available(&self) -> bool {
        true
    }

    fn chunk_data(
        &self,
        data: &[u8],
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Vec<(String, usize, usize)>, ChunkingError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuFastCdcBackend;

impl RollingHashBackend for CpuFastCdcBackend {
    fn name(&self) -> &'static str {
        "cpu-fastcdc"
    }

    fn chunk_data(
        &self,
        data: &[u8],
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
        chunk_data_cpu(data, min_size, avg_size, max_size)
    }
}

/// Dispatch to the chosen backend. CPU FastCDC remains the default, while the GPU backend
/// is used when compiled with the `gpu` feature and a compatible device is found at runtime.
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    chunk_data_with_backend(ChunkingBackend::Cpu, data, min_size, avg_size, max_size)
}

pub fn chunk_data_with_backend(
    backend: ChunkingBackend,
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    match backend {
        ChunkingBackend::Cpu => CpuFastCdcBackend.chunk_data(data, min_size, avg_size, max_size),
        ChunkingBackend::Auto => {
            #[cfg(feature = "gpu")]
            {
                if let Some(gpu) = gpu::GpuRollingHashBackend::new() {
                    return gpu.chunk_data(data, min_size, avg_size, max_size);
                }
            }

            CpuFastCdcBackend.chunk_data(data, min_size, avg_size, max_size)
        }
    }
}

/// Chunk data using `FastCDC` (Content-Defined Chunking)
/// Args: data (binary), `min_size` (optional), `avg_size` (optional), `max_size` (optional)
/// Returns: list of {`chunk_hash`, `offset`, `length`}
fn chunk_data_cpu(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    // These values are well below u32::MAX, so truncation is safe
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32; // 256 KB

    let chunker = FastCDC::new(data, min, avg, max);

    let mut chunks = Vec::new();

    for chunk in chunker {
        // Validate bounds before slice access (defense-in-depth)
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        chunks.push((hash_hex, chunk.offset, chunk.length));
    }

    Ok(chunks)
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{validate_slice_bounds, ChunkingError, RollingHashBackend};
    use bytemuck::{Pod, Zeroable};
    use sha2::{Digest, Sha256};
    use wgpu::{self, util::DeviceExt};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    struct Params {
        len: u32,
        min: u32,
        max: u32,
        mask: u32,
    }

    #[derive(Clone)]
    pub struct GpuRollingHashBackend {
        adapter: wgpu::Adapter,
        adapter_info: wgpu::AdapterInfo,
    }

    impl GpuRollingHashBackend {
        pub fn new() -> Option<Self> {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))?;

            let adapter_info = adapter.get_info();
            match adapter_info.device_type {
                wgpu::DeviceType::DiscreteGpu
                | wgpu::DeviceType::IntegratedGpu
                | wgpu::DeviceType::VirtualGpu => Some(Self {
                    adapter,
                    adapter_info,
                }),
                _ => None,
            }
        }

        fn gear_mask(avg: u32) -> u32 {
            let next_pow = avg.next_power_of_two();
            next_pow.saturating_sub(1)
        }

        fn build_shader() -> wgpu::ShaderModuleDescriptor<'static> {
            const SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> boundaries: array<u32>;

@group(0) @binding(2)
var<storage, read_write> counter: array<atomic<u32>>;

struct Params {
    len: u32,
    min: u32,
    max: u32,
    mask: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    var pos: u32 = 0u;
    var rolling: u32 = 0u;

    loop {
        if (pos >= params.len) {
            break;
        }

        let byte = data[pos];
        rolling = (rolling << 5u) ^ (rolling >> 2u) ^ byte;

        let at_least_min = pos + 1u >= params.min;
        let hit_max = pos + 1u >= params.max;
        let boundary = (at_least_min && ((rolling & params.mask) == 0u)) || hit_max;

        if (boundary) {
            let idx = atomicAdd(&counter[0], 1u);
            boundaries[idx] = pos + 1u;
            rolling = 0u;
        }

        pos = pos + 1u;
    }
}
"#;

            wgpu::ShaderModuleDescriptor {
                label: Some("chunk-boundary-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            }
        }
    }

    impl RollingHashBackend for GpuRollingHashBackend {
        fn name(&self) -> &'static str {
            "gpu-rolling-hash"
        }

        fn is_available(&self) -> bool {
            matches!(
                self.adapter_info.device_type,
                wgpu::DeviceType::DiscreteGpu
                    | wgpu::DeviceType::IntegratedGpu
                    | wgpu::DeviceType::VirtualGpu
            )
        }

        #[allow(clippy::too_many_lines)]
        fn chunk_data(
            &self,
            data: &[u8],
            min_size: Option<usize>,
            avg_size: Option<usize>,
            max_size: Option<usize>,
        ) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
            if data.is_empty() {
                return Ok(Vec::new());
            }

            let min = min_size.unwrap_or(16_384) as u32;
            let avg = avg_size.unwrap_or(65_536) as u32;
            let max = max_size.unwrap_or(262_144) as u32;

            let (device, queue) = pollster::block_on(
                self.adapter
                    .request_device(&wgpu::DeviceDescriptor::default(), None),
            )
            .map_err(|_| ChunkingError::GpuUnavailable {
                reason: "device_request_failed",
            })?;

            let params = Params {
                len: data.len() as u32,
                min,
                max,
                mask: Self::gear_mask(avg),
            };

            let data_u32: Vec<u32> = data.iter().copied().map(u32::from).collect();
            let capacity = usize::max(1, data.len() / usize::max(1, min as usize) + 2);
            let zero_counter: [u32; 1] = [0];

            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("data"),
                contents: bytemuck::cast_slice(&data_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let boundary_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boundaries"),
                size: (capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("counter"),
                contents: bytemuck::bytes_of(&zero_counter),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("chunking-pipeline-layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

            let shader = device.create_shader_module(Self::build_shader());
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chunking-compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunking-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: boundary_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: counter_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunking-encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("chunking-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            let count_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("count-readback"),
                size: std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let boundary_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boundary-readback"),
                size: (capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(
                &counter_buffer,
                0,
                &count_readback,
                0,
                std::mem::size_of::<u32>() as u64,
            );
            encoder.copy_buffer_to_buffer(
                &boundary_buffer,
                0,
                &boundary_readback,
                0,
                (capacity * std::mem::size_of::<u32>()) as u64,
            );

            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);

            let count_slice = count_readback.slice(..);
            let boundary_slice = boundary_readback.slice(..);

            let (count_tx, count_rx) = futures_intrusive::channel::shared::oneshot_channel();
            count_slice.map_async(wgpu::MapMode::Read, move |v| {
                count_tx.send(v).ok();
            });

            let (bound_tx, bound_rx) = futures_intrusive::channel::shared::oneshot_channel();
            boundary_slice.map_async(wgpu::MapMode::Read, move |v| {
                bound_tx.send(v).ok();
            });

            pollster::block_on(async {
                let _ = count_rx.receive().await;
                let _ = bound_rx.receive().await;
            });

            let count_data: Vec<u32> = {
                let view = count_slice.get_mapped_range();
                let mut vec = Vec::new();
                vec.extend_from_slice(bytemuck::cast_slice(&view));
                drop(view);
                vec
            };

            let boundary_data: Vec<u32> = {
                let view = boundary_slice.get_mapped_range();
                let mut vec = Vec::new();
                vec.extend_from_slice(bytemuck::cast_slice(&view));
                drop(view);
                vec
            };

            count_readback.unmap();
            boundary_readback.unmap();

            let chunk_count = usize::try_from(*count_data.get(0).unwrap_or(&0)).unwrap_or(0);
            let mut offsets = boundary_data;
            offsets.truncate(chunk_count);

            if offsets.last().copied().unwrap_or(0) != data.len() as u32 {
                offsets.push(data.len() as u32);
            }

            let mut chunks = Vec::new();
            let mut start: usize = 0;
            for &end in &offsets {
                let end_usize = usize::try_from(end).unwrap_or(data.len());
                validate_slice_bounds(data.len(), start, end_usize.saturating_sub(start))?;
                let mut hasher = Sha256::new();
                hasher.update(&data[start..end_usize]);
                let hash_hex = hex::encode(hasher.finalize());
                chunks.push((hash_hex, start, end_usize - start));
                start = end_usize;
            }

            Ok(chunks)
        }
    }
}
