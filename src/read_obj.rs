use tobj::{self};

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

    triangles.extend_from_slice(&(-4.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    // material
    triangles.extend_from_slice(&(0_f32).to_le_bytes());

    triangles.extend_from_slice(&4.0_f32.to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    // pad
    triangles.extend_from_slice(&(0.2_f32).to_le_bytes());

    triangles.extend_from_slice(&(-4.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-6.0_f32).to_le_bytes());
    // pad
    triangles.extend_from_slice(&(0.0_f32).to_le_bytes());

    //triangle 2 really
    triangles.extend_from_slice(&(4.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-6.0_f32).to_le_bytes());
    // material
    triangles.extend_from_slice(&(1.0_f32).to_le_bytes());

    triangles.extend_from_slice(&4.0_f32.to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    // pad
    triangles.extend_from_slice(&(0.2_f32).to_le_bytes());

    triangles.extend_from_slice(&(-4.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-1.0_f32).to_le_bytes());
    triangles.extend_from_slice(&(-6.0_f32).to_le_bytes());
    // pad
    triangles.extend_from_slice(&(1.0_f32).to_le_bytes());

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
            triangles.extend_from_slice(&2.0_f32.to_le_bytes());

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