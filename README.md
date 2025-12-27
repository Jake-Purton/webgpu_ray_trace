# Porting Ray Tracing to Rust and WebGPU

## Introduction

In this post, I’ll walk through my journey of extending the [Ray Tracing Road to Rust (RTRR)](https://the-ray-tracing-road-to-rust.vercel.app/6-surface-normals-and-multiple-objects) project. I’ll cover how I added triangle support, imported geometry from OBJ files, and started porting the project to WebGPU. My project used `minifb` whereas RTRR wrote out PPM image files. This is the only major difference.

---

## Adding Triangles to the CPU Version

RTRR is a fantastic project for beginners learning rust or ray tracing. It focuses mainly on spheres and I wanted to add triangles. The specific working commit is [here](https://github.com/Jake-Purton/webgpu_ray_trace/tree/b26dc46eb08a39acbcfbff3ad3bbe284c25d8d4d) in my repository. Please note that my obj file reading wasn't correct at the time but it was close enough to see some triangles.

**The Triangle Struct**
```rust
// triangle.rs
use std::sync::Arc;

use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{self, Point3, unit_vector};

pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
    pub mat: Arc<dyn Material + Send + Sync>,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3, m: Arc<dyn Material + Send + Sync>) -> Triangle {
        Triangle {
            a,
            b,
            c,
            mat: m,
        }
    }
}

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord) -> bool {
        // Möller–Trumbore intersection algorithm
        let epsilon = 1e-8;
        let edge1 = self.b - self.a;
        let edge2 = self.c - self.a;
        let h = vec3::cross(r.direction(), edge2);
        let a = vec3::dot(edge1, h);

        if a.abs() < epsilon {
            return false; // Ray is parallel to triangle
        }

        let f = 1.0 / a;
        let s = r.origin() - self.a;
        let u = f * vec3::dot(s, h);

        if u < 0.0 || u > 1.0 {
            return false;
        }

        let q = vec3::cross(s, edge1);
        let v = f * vec3::dot(r.direction(), q);

        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t = f * vec3::dot(edge2, q);

        if t < t_min || t > t_max {
            return false;
        }

        rec.t = t;
        rec.p = r.at(t);
        let outward_normal = unit_vector(vec3::cross(edge1, edge2));
        rec.set_face_normal(r, outward_normal);
        rec.mat = Some(self.mat.clone());
        true
    }
}
```

This implements the tutorial's `Hittable` trait and can therefore be added to world like so.

```rust
    world.add(Box::new(Triangle::new(
        Point3::new(-4.0_f32, -1.0_f32, -1.0_f32),
        Point3::new(4.0_f32, -1.0_f32, -1.0_f32),
        Point3::new(-4.0_f32, -1.0_f32, -6.0_f32),
        material_ground.clone(),
    )));
```

---

## Reading Triangles from OBJ Files

To render real models, I used `tobj` to load triangles from OBJ files. Here is an example loading the file 'suzanne.obj'.

```rust
let (models, _) = tobj::load_obj(
    "suzanne.obj",
    &tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    },
)
.unwrap();

for model in models {
    let mesh = &model.mesh;
    let positions = &mesh.positions;
    let indices = &mesh.indices;

    for i in (0..indices.len()).step_by(3) {
        let i0 = indices[i] as usize * 3;
        let i1 = indices[i + 1] as usize * 3;
        let i2 = indices[i + 2] as usize * 3;

        // the points are in:
        // positions[i0], positions[i0+1], positions[i0+2]
        // positions[i1], positions[i1+1], positions[i1+2]
        // positions[i2], positions[i2+1], positions[i2+2]
        // and can be turned into Triangle objects and passed into World

    }
}
```

---

## Preparing for WebGPU

Currently, with ray tracing running on the CPU the simulation is slow even on relatively high end hardware. We can get a huge boost in speed by sending our triangles to the GPU to have the simulation run in paralel there. This will involve sending our triangles in a buffer, implementing a ray tracing shader, and recieving the results as pixels in another buffer before displaying this to the screen.

---

### Input and Output Buffers for Triangles

Here is what the setup looks like for sending triangles to the GPU. In the loop where we read the triangles from the obj file before, we now populate a `Vec<u8>`. Our goal is to send 3 sets of 3 `f32`s, however because of how wgsl reads buffers we actually need 4 bytes of padding between each point in the triangle.

```rust
let mut triangles: Vec<u8> = Vec::new();
let suzanne_offset = -2.5;
for model in models {
    let mesh = &model.mesh;
    let positions = &mesh.positions;
    let indices = &mesh.indices;

    for i in (0..indices.len()).step_by(3) {
        let i0 = indices[i] as usize * 3;
        let i1 = indices[i + 1] as usize * 3;
        let i2 = indices[i + 2] as usize * 3;

        // vertex 1
        triangles.extend_from_slice(&positions[i0].to_le_bytes());
        triangles.extend_from_slice(&positions[i0 + 1].to_le_bytes());
        triangles.extend_from_slice(&(positions[i0 + 2] + suzanne_offset).to_le_bytes());
        // pad
        triangles.extend_from_slice(&0.0_f32.to_le_bytes());

        // vertex 2
        triangles.extend_from_slice(&positions[i1].to_le_bytes());
        triangles.extend_from_slice(&positions[i1 + 1].to_le_bytes());
        triangles.extend_from_slice(&(positions[i1 + 2] + suzanne_offset).to_le_bytes());
        // pad
        triangles.extend_from_slice(&0.0_f32.to_le_bytes());

        // vertex 3
        triangles.extend_from_slice(&positions[i2].to_le_bytes());
        triangles.extend_from_slice(&positions[i2 + 1].to_le_bytes());
        triangles.extend_from_slice(&(positions[i2 + 2] + suzanne_offset).to_le_bytes());
        // pad
        triangles.extend_from_slice(&0.0_f32.to_le_bytes());
    }
}
```

---

### Buffer Management Lessons

- **Unused Buffers Are Removed:** I discovered that if a buffer isn’t used in a shader, it may be removed by the pipeline.
- **Buffer Padding:** WebGPU expects data in a specific byte layout, so I had to pad triangle structs to match alignment requirements.

**Code Example: Buffer Padding**
```rust
// Show how you pad structs for GPU alignment here
```

---

### Camera and Material Buffers

I added a parameters buffer for the camera and a materials buffer. Material data is now stored in the triangle’s `a.w` field.

**Code Example: Camera/Material Buffer**
```rust
// Show how you set up and use these buffers here
```

---

## Conclusion

todo

---

## References

- [Ray Tracing Road to Rust](https://the-ray-tracing-road-to-rust.vercel.app/6-surface-normals-and-multiple-objects)
- [Sebastian Lague - Coding Adventures: Ray Tracing ](https://www.youtube.com/watch?v=Qz0KTGYJtUk)

---

## The future

This project is nowhere close to done. I'd like to keep working on it and these are th things I would work on next. Please consider forking [the repository](https://github.com/Jake-Purton/webgpu_ray_trace) and adding any of these features. If you do add any then also add a markdown file with a little detail about your journey into coming up with the solution.
- modularity of wgsl using string concatenation
- add metallic and glass etc materials
- movable camera
- recompute each frame
- add BVH for optimisation