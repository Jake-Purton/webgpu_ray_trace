#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    origin: [f32; 4],
    lower_left_corner: [f32; 4],
    horizontal: [f32; 4],
    vertical: [f32; 4],
}

impl Camera {
    pub fn new() -> Camera {
        let aspect_ratio = 16.0 / 9.0;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;

        let origin = [0.0, 0.0, 0.0, 32.0];
        let horizontal = [viewport_width, 0.0, 0.0, 64.0];
        let vertical = [0.0, viewport_height, 0.0, 64.0];

        let lower_left_corner: [f32; 4] = [
            origin[0] - horizontal[0] / 2.0 - vertical[0] / 2.0 - 0.0,
            origin[1] - horizontal[1] / 2.0 - vertical[1] / 2.0 - 0.0,
            origin[2] - horizontal[2] / 2.0 - vertical[2] / 2.0 - focal_length,
            64.0,
        ];

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }
}