#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    emission: [f32; 4],
    albedo: [f32; 4],
    material_type: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

impl Material {

    pub fn new(emission: [f32;4], albedo: [f32;4], material_type: u32) -> Self {

        Self { emission, albedo, material_type, pad1: 0, pad2: 0, pad3: 0 }

    }

    pub fn list() -> Vec<u8> {
        let mut vec = Vec::new();

        // 0 gray
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0],
            1,
        )));

        // 1 red
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.99, 0.2, 0.2, 0.0],
            1,
        )));

        // 2 blue
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.99, 0.0],
            1,
        )));

        // 3 green
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.99, 0.2, 0.0],
            1,
        )));

        // 4 white
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.99, 0.99, 0.99, 0.0],
            1,
        )));

        // 5 light
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.99, 0.99, 0.99, 0.99],
            [0.99, 0.99, 0.99, 0.0],
            1,
        )));
        return vec;
    }
}
