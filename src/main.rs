mod camera;
mod materials;
mod read_obj;

use camera::Camera;
use minifb::{Key, Window, WindowOptions};

use bytemuck;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Instance, InstanceDescriptor};

use crate::materials::Material;
use crate::read_obj::read_obj_vertices;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
    camera: Camera,
    depth: u32,
    samples: u32,
    _pad5: u32,
    _pad6: u32,
}

const WIDTH: usize = 1080;
const HEIGHT: usize = 720;
const SAMPLES_PER_PIXEL: u32 = 8;
const MAX_DEPTH: u32 = 4;

fn main() {
    let v = read_obj_vertices("suzanne.obj");

    let instance = Instance::new(&InstanceDescriptor {
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));

    let adapter = match adapter {
        Ok(a) => {
            println!("Adapter found: {:?}", a.get_info().name);
            a
        }
        Err(_) => {
            println!("ERROR: No GPU adapter found. WebGPU may not be supported in this browser.");
            return;
        }
    };

    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
            Ok(a) => a,
            Err(e) => {
                println!("{e}");
                return;
            }
        };

    let input_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: &v,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
    });

    let materials_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Materials Buffer"),
        contents: &Material::list(),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = Params {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        _pad1: 0,
        _pad2: 0,
        camera: Camera::new(),
        depth: MAX_DEPTH,
        samples: SAMPLES_PER_PIXEL,
        _pad5: 0,
        _pad6: 0,
    };

    let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output_size = (WIDTH * HEIGHT) * std::mem::size_of::<u32>();

    // output buffer in gpu memory
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // read the output into cpu memory
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Tracing Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("trace.wgsl").into()),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: materials_buffer.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        cpass.dispatch_workgroups(
            ((WIDTH + 7) / 8).try_into().unwrap(),
            ((HEIGHT + 7) / 8).try_into().unwrap(),
            1,
        ); // example workgroup
    }

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        Some(output_size as u64),
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("hello");

    let data = buffer_slice.get_mapped_range();
    let result: &[u32] = bytemuck::cast_slice(&data);

    let mut window = Window::new("Rustracer", WIDTH, HEIGHT, WindowOptions::default())
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&result, WIDTH, HEIGHT).unwrap();
    }
}
