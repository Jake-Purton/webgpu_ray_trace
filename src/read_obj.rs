use tobj::{self};

pub fn square_to_bytes(a: [f32; 3], b: [f32; 3], material: f32) -> Vec<u8> {
    let (min_x, max_x) = if a[0] < b[0] { (a[0], b[0]) } else { (b[0], a[0]) };
    let (min_y, max_y) = if a[1] < b[1] { (a[1], b[1]) } else { (b[1], a[1]) };
    let (min_z, max_z) = if a[2] < b[2] { (a[2], b[2]) } else { (b[2], a[2]) };

    // Determine which axis is constant (the two input points must differ in exactly two axes)
    let dx = (a[0] - b[0]).abs();
    let dy = (a[1] - b[1]).abs();
    let dz = (a[2] - b[2]).abs();

    let (v0, v1, v2, v3) = if dx < 1e-6 {
        // x is constant, square in YZ plane
        (
            [a[0], min_y, min_z],
            [a[0], max_y, min_z],
            [a[0], max_y, max_z],
            [a[0], min_y, max_z],
        )
    } else if dy < 1e-6 {
        // y is constant, square in XZ plane
        (
            [min_x, a[1], min_z],
            [max_x, a[1], min_z],
            [max_x, a[1], max_z],
            [min_x, a[1], max_z],
        )
    } else {
        // z is constant, square in XY plane
        (
            [min_x, min_y, a[2]],
            [max_x, min_y, a[2]],
            [max_x, max_y, a[2]],
            [min_x, max_y, a[2]],
        )
    };

    let mut bytes = Vec::new();

    // Triangle 1: v0, v1, v2
    for v in [v0, v1, v2] {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
        bytes.extend_from_slice(&material.to_le_bytes());
    }
    // Triangle 2: v0, v2, v3
    for v in [v0, v2, v3] {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
        bytes.extend_from_slice(&material.to_le_bytes());
    }

    bytes
}

pub fn read_obj_vertices(filename: &str) -> Vec<u8> {
    let (models, _) = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .unwrap();

    let mut triangles: Vec<u8> = Vec::new();
    let suzanne_offset = -2.5;

    // floor square
    triangles.extend_from_slice(&square_to_bytes([-4.0, -1.0, -1.0], [4.0, -1.0, -6.0], 0.0));

    // Back wall
    triangles.extend_from_slice(&square_to_bytes([-4.0, 3.0, -6.0], [4.0, -1.0, -6.0], 1.0));

    // Left wall
    triangles.extend_from_slice(&square_to_bytes([-4.0, 3.0, -1.0], [-4.0, -1.0, -6.0], 2.0));

    // Right wall
    triangles.extend_from_slice(&square_to_bytes([4.0, 3.0, -1.0], [4.0, -1.0, -6.0], 3.0));

    // top
    triangles.extend_from_slice(&square_to_bytes([-4.0, 3.0, -1.0], [4.0, 3.0, -6.0], 5.0));

    // return triangles;

    for model in models {
        let mesh = &model.mesh;
        let positions = &mesh.positions;
        let indices = &mesh.indices;

        for i in (0..indices.len()).step_by(3) {
            let i0 = indices[i] as usize * 3;
            let i1 = indices[i + 1] as usize * 3;
            let i2 = indices[i + 2] as usize * 3;

            // f32s to bytes little endian

            // this all represents 1 triangle
            triangles.extend_from_slice(&positions[i0].to_le_bytes());
            triangles.extend_from_slice(&positions[i0 + 1].to_le_bytes());
            triangles.extend_from_slice(&(positions[i0 + 2] + suzanne_offset).to_le_bytes());
            // material
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