mod camera;
mod materials;
mod read_obj;

use camera::Camera;
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

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
    frame: u32,
    _pad6: u32,
}

const WIDTH: usize = 400;
const HEIGHT: usize = 225;
const SAMPLES_PER_PIXEL: u32 = 16;
const MAX_DEPTH: u32 = 4;
const TRIANGLE_STRIDE_BYTES: usize = 12 * std::mem::size_of::<f32>();
const MONKEY_MATERIAL_ID: f32 = 4.0;
const CAMERA_ORBIT_RADIUS: f32 = 3.5;
const CAMERA_ORBIT_HEIGHT: f32 = 0.4;
const CAMERA_ORBIT_SPEED_RAD_PER_SEC: f32 = 0.7;

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    f32::from_le_bytes(raw)
}

fn suzanne_center(scene_bytes: &[u8]) -> Option<[f32; 3]> {
    let mut sum = [0.0_f32; 3];
    let mut count: u32 = 0;

    for tri_start in (0..scene_bytes.len()).step_by(TRIANGLE_STRIDE_BYTES) {
        let material = read_f32_le(scene_bytes, tri_start + 3 * 4);
        if (material - MONKEY_MATERIAL_ID).abs() > f32::EPSILON {
            continue;
        }

        for vertex_base in [tri_start, tri_start + 4 * 4, tri_start + 8 * 4] {
            sum[0] += read_f32_le(scene_bytes, vertex_base);
            sum[1] += read_f32_le(scene_bytes, vertex_base + 4);
            sum[2] += read_f32_le(scene_bytes, vertex_base + 8);
            count = count.wrapping_add(1);
        }
    }

    if count == 0 {
        None
    } else {
        Some([
            sum[0] / count as f32,
            sum[1] / count as f32,
            sum[2] / count as f32,
        ])
    }
}

fn main() {
    let scene_bytes = read_obj_vertices("suzanne.obj");
    let monkey_center = match suzanne_center(&scene_bytes) {
        Some(center) => center,
        None => {
            println!("ERROR: Could not determine Suzanne center for camera orbit.");
            return;
        }
    };

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
        contents: &scene_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let materials_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Materials Buffer"),
        contents: &Material::list(),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut params = Params {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        _pad1: 0,
        _pad2: 0,
        camera: Camera::new(),
        depth: MAX_DEPTH,
        samples: SAMPLES_PER_PIXEL,
        frame: 0,
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

    let mut window = Window::new("Rustracer", WIDTH, HEIGHT, WindowOptions::default())
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

    let mut frame_buffer = vec![0u32; WIDTH * HEIGHT];
    let mut frames: u32 = 0;
    let mut fps_timer = Instant::now();
    let camera_timer = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let orbit_angle = camera_timer.elapsed().as_secs_f32() * CAMERA_ORBIT_SPEED_RAD_PER_SEC;
        params.camera = Camera::orbit_y(
            monkey_center,
            CAMERA_ORBIT_RADIUS,
            CAMERA_ORBIT_HEIGHT,
            orbit_angle,
            WIDTH as f32 / HEIGHT as f32,
        );

        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

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
            );
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
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");

        rx.recv()
            .expect("Map callback channel closed")
            .expect("Failed to map staging buffer for reading");

        {
            let data = buffer_slice.get_mapped_range();
            let result: &[u32] = bytemuck::cast_slice(&data);
            frame_buffer.copy_from_slice(result);
        }
        staging_buffer.unmap();

        window.update_with_buffer(&frame_buffer, WIDTH, HEIGHT).unwrap();

        params.frame = params.frame.wrapping_add(1);
        frames = frames.wrapping_add(1);

        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let fps = (frames as f32) / elapsed;
            window.set_title(&format!("Rustracer - {:.1} FPS", fps));
            frames = 0;
            fps_timer = Instant::now();
        }
    }
}
