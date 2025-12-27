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

        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            1,
        )));

        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.3, 0.2, 0.0],
            2,
        )));

        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.2, 0.0],
            3,
        )));
        vec.extend_from_slice(bytemuck::bytes_of(&Self::new(
            [0.8, 0.3, 0.6, 0.8],
            [0.0, 1.0, 1.0, 0.0],
            4,
        )));

        return vec;
    }
}
