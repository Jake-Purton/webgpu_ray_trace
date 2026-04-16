#[cfg(target_arch = "wasm32")]
use std::io::BufReader;
use tobj::{self};

pub fn square_to_bytes(a: [f32; 3], b: [f32; 3], material: f32) -> Vec<u8> {
    let (min_x, max_x) = if a[0] < b[0] { (a[0], b[0]) } else { (b[0], a[0]) };
    let (min_y, max_y) = if a[1] < b[1] { (a[1], b[1]) } else { (b[1], a[1]) };
    let (min_z, max_z) = if a[2] < b[2] { (a[2], b[2]) } else { (b[2], a[2]) };

    let dx = (a[0] - b[0]).abs();
    let dy = (a[1] - b[1]).abs();

    let (v0, v1, v2, v3) = if dx < 1e-6 {
        (
            [a[0], min_y, min_z],
            [a[0], max_y, min_z],
            [a[0], max_y, max_z],
            [a[0], min_y, max_z],
        )
    } else if dy < 1e-6 {
        (
            [min_x, a[1], min_z],
            [max_x, a[1], min_z],
            [max_x, a[1], max_z],
            [min_x, a[1], max_z],
        )
    } else {
        (
            [min_x, min_y, a[2]],
            [max_x, min_y, a[2]],
            [max_x, max_y, a[2]],
            [min_x, max_y, a[2]],
        )
    };

    let mut bytes = Vec::new();

    for v in [v0, v1, v2] {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
        bytes.extend_from_slice(&material.to_le_bytes());
    }

    for v in [v0, v2, v3] {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
        bytes.extend_from_slice(&material.to_le_bytes());
    }

    bytes
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_obj_vertices(filename: &str) -> Vec<u8> {
    let (models, _) = tobj::load_obj(filename, &obj_load_options()).unwrap();
    build_scene_from_models(models)
}

#[cfg(target_arch = "wasm32")]
pub fn read_obj_vertices_from_bytes(obj_bytes: &[u8]) -> Vec<u8> {
    let mut reader = BufReader::new(obj_bytes);
    let (models, _) = tobj::load_obj_buf(&mut reader, &obj_load_options(), |_| {
        Ok((Vec::new(), Default::default()))
    })
    .unwrap();

    build_scene_from_models(models)
}

fn obj_load_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    }
}

fn build_scene_from_models(models: Vec<tobj::Model>) -> Vec<u8> {
    let mut triangles: Vec<u8> = Vec::new();
    let suzanne_offset = -2.5;

    let room_min_x = -4.0;
    let room_max_x = 4.0;
    let room_floor_y = -1.0;
    let room_ceiling_y = 3.0;
    let room_back_z = -6.0;
    let room_front_z = 2.0;

    triangles.extend_from_slice(&square_to_bytes(
        [room_min_x, room_floor_y, room_front_z],
        [room_max_x, room_floor_y, room_back_z],
        0.0,
    ));

    triangles.extend_from_slice(&square_to_bytes(
        [room_min_x, room_ceiling_y, room_back_z],
        [room_max_x, room_floor_y, room_back_z],
        1.0,
    ));

    triangles.extend_from_slice(&square_to_bytes(
        [room_min_x, room_ceiling_y, room_front_z],
        [room_max_x, room_floor_y, room_front_z],
        0.0,
    ));

    triangles.extend_from_slice(&square_to_bytes(
        [room_min_x, room_ceiling_y, room_front_z],
        [room_min_x, room_floor_y, room_back_z],
        2.0,
    ));

    triangles.extend_from_slice(&square_to_bytes(
        [room_max_x, room_ceiling_y, room_front_z],
        [room_max_x, room_floor_y, room_back_z],
        3.0,
    ));

    triangles.extend_from_slice(&square_to_bytes(
        [room_min_x, room_ceiling_y, room_front_z],
        [room_max_x, room_ceiling_y, room_back_z],
        5.0,
    ));

    for model in models {
        let mesh = &model.mesh;
        let positions = &mesh.positions;
        let indices = &mesh.indices;

        for i in (0..indices.len()).step_by(3) {
            let i0 = indices[i] as usize * 3;
            let i1 = indices[i + 1] as usize * 3;
            let i2 = indices[i + 2] as usize * 3;

            triangles.extend_from_slice(&positions[i0].to_le_bytes());
            triangles.extend_from_slice(&positions[i0 + 1].to_le_bytes());
            triangles.extend_from_slice(&(positions[i0 + 2] + suzanne_offset).to_le_bytes());
            triangles.extend_from_slice(&4.0_f32.to_le_bytes());

            triangles.extend_from_slice(&positions[i1].to_le_bytes());
            triangles.extend_from_slice(&positions[i1 + 1].to_le_bytes());
            triangles.extend_from_slice(&(positions[i1 + 2] + suzanne_offset).to_le_bytes());
            triangles.extend_from_slice(&0.0_f32.to_le_bytes());

            triangles.extend_from_slice(&positions[i2].to_le_bytes());
            triangles.extend_from_slice(&positions[i2 + 1].to_le_bytes());
            triangles.extend_from_slice(&(positions[i2 + 2] + suzanne_offset).to_le_bytes());
            triangles.extend_from_slice(&0.0_f32.to_le_bytes());
        }
    }

    triangles
}
