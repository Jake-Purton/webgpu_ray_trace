use minifb::{Key, Window, WindowOptions};
use std::sync::{Arc, Mutex};
use std::thread;

mod camera;
mod colour;
mod common;
mod hittable;
mod hittable_list;
mod ray;
mod sphere;
mod vec3;
mod material;

use camera::Camera;
use colour::Colour;
use hittable::{HitRecord, Hittable};
use hittable_list::HittableList;
use ray::Ray;
use sphere::Sphere;
use vec3::Point3;
use material::{Lambertian, Metal};


const WIDTH: usize = 400;
const HEIGHT: usize = 225;
const SAMPLES_PER_PIXEL: i32 = 5;
const MAX_DEPTH: i32 = 3;
 
fn ray_color(r: &Ray, world: &dyn Hittable, depth: i32) -> Colour {
    // If we've exceeded the ray bounce limit, no more light is gathered
    if depth <= 0 {
        return Colour::new(0.0, 0.0, 0.0);
    }
    let mut rec = HitRecord::new();
    if world.hit(r, 0.001, common::INFINITY, &mut rec) {
        let mut attenuation = Colour::default();
        let mut scattered = Ray::default();
        if rec
            .mat
            .as_ref()
            .unwrap()
            .scatter(r, &rec, &mut attenuation, &mut scattered)
        {
            return attenuation * ray_color(&scattered, world, depth - 1);
        }
        return Colour::new(0.0, 0.0, 0.0);
    }

    let unit_direction = vec3::unit_vector(r.direction());
    let t = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - t) * Colour::new(1.0, 1.0, 1.0) + t * Colour::new(0.5, 0.7, 1.0)
}

fn main() {
    // Create a shared buffer for the pixel data
    let buffer = Arc::new(Mutex::new(vec![0u32; WIDTH * HEIGHT]));

 
    let mut world = HittableList::new();
    
    let material_ground = Arc::new(Lambertian::new(Colour::new(0.8, 0.8, 0.0)));
    let material_center = Arc::new(Lambertian::new(Colour::new(0.7, 0.3, 0.3)));
    let material_left = Arc::new(Metal::new(Colour::new(0.8, 0.8, 0.8), 0.3));
    let material_right = Arc::new(Metal::new(Colour::new(0.8, 0.6, 0.2), 1.0));
 
    world.add(Box::new(Sphere::new(
        Point3::new(0.0, -100.5, -1.0),
        100.0,
        material_ground,
    )));
    world.add(Box::new(Sphere::new(
        Point3::new(0.0, 0.0, -1.0),
        0.5,
        material_center,
    )));
    world.add(Box::new(Sphere::new(
        Point3::new(-1.0, 0.0, -1.0),
        0.5,
        material_left,
    )));
    world.add(Box::new(Sphere::new(
        Point3::new(1.0, 0.0, -1.0),
        0.5,
        material_right,
    )));
    let world = Arc::new(world);
    // Camera
 
    let cam = Arc::new(Camera::new());

    // Create the window
    let mut window = Window::new(
        "Rustracer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Number of worker threads (reasonable number, not one per pixel/row)
    let num_threads = 8;
    let rows_per_thread = HEIGHT / num_threads;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut handles = vec![];

        // Spawn worker threads
        for thread_id in 0..num_threads {
            let buffer = Arc::clone(&buffer);
            let world = Arc::clone(&world);
            let cam = Arc::clone(&cam);
            let start_row = thread_id * rows_per_thread;
            let end_row = if thread_id == num_threads - 1 {
                HEIGHT
            } else {
                (thread_id + 1) * rows_per_thread
            };

            let handle = thread::spawn(move || {
                let mut buffer = buffer.lock().unwrap();

                for i in start_row..end_row {
                    for j in 0..WIDTH {
                        let index = (HEIGHT - i - 1) * WIDTH + j;

                        let mut pixel_color = Colour::new(0.0, 0.0, 0.0);

                        for _ in 0..SAMPLES_PER_PIXEL {
                            let u = (j as f64 + common::random_double()) / (WIDTH - 1) as f64;
                            let v = (i as f64 + common::random_double()) / (HEIGHT - 1) as f64;
                            let r = cam.get_ray(u, v);
                            pixel_color += ray_color(&r, &*world, MAX_DEPTH);
                        }

                        let scale = 1.0 / SAMPLES_PER_PIXEL as f64;

                        pixel_color *= scale;
                        pixel_color.sqrt_each();
                        
                        // Pack RGB into u32 (minifb format: 0x00RRGGBB)
                        buffer[index] = pixel_color.to_u32();
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Update the window with the buffer
        let buffer = buffer.lock().unwrap();
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}