use std::sync::Arc;

use tobj::{self};
use crate::{colour::Colour, material::Lambertian, triangle::Triangle, vec3::Vec3};

pub fn read_obj_vertices(filename: &str) -> Vec<Triangle> {
    let (models, _) = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .unwrap();

    let mut triangles: Vec<Triangle> = Vec::new();
    let suzanne_offset = -2.5;

    let material = Arc::new(Lambertian::new(Colour::new(0.8, 0.6, 0.8)));

    for model in models {
        let mesh = &model.mesh;
        let positions = &mesh.positions;
        let indices = &mesh.indices;

        for i in (0..indices.len()).step_by(3) {
            let i0 = indices[i] as usize * 3;
            let i1 = indices[i + 1] as usize * 3;
            let i2 = indices[i + 2] as usize * 3;

            triangles.push(Triangle {
                a: Vec3::new(positions[i0], positions[i0 + 1], positions[i0 + 2] + suzanne_offset),
                b: Vec3::new(positions[i1], positions[i1 + 1], positions[i1 + 2] + suzanne_offset),
                c: Vec3::new(positions[i2], positions[i2 + 1], positions[i2 + 2] + suzanne_offset),
                mat: material.clone()
            });
        }
    }

    triangles
}