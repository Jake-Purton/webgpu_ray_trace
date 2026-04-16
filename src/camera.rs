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
        Self::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -2.5], [0.0, 1.0, 0.0], 60.0, 16.0 / 9.0)
    }

    pub fn orbit_y(
        target: [f32; 3],
        radius: f32,
        height: f32,
        angle_rad: f32,
        aspect_ratio: f32,
    ) -> Camera {
        let origin = [
            target[0] + radius * angle_rad.cos(),
            target[1] + height,
            target[2] + radius * angle_rad.sin(),
        ];

        Self::look_at(origin, target, [0.0, 1.0, 0.0], 60.0, aspect_ratio)
    }

    fn look_at(
        origin: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        vfov_degrees: f32,
        aspect_ratio: f32,
    ) -> Camera {
        let theta = vfov_degrees.to_radians();
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = normalize(sub(origin, target));
        let u = normalize(cross(up, w));
        let v = cross(w, u);

        let horizontal = scale(u, viewport_width);
        let vertical = scale(v, viewport_height);
        let lower_left = sub(sub(sub(origin, scale(horizontal, 0.5)), scale(vertical, 0.5)), w);

        Camera {
            origin: [origin[0], origin[1], origin[2], 0.0],
            lower_left_corner: [lower_left[0], lower_left[1], lower_left[2], 0.0],
            horizontal: [horizontal[0], horizontal[1], horizontal[2], 0.0],
            vertical: [vertical[0], vertical[1], vertical[2], 0.0],
        }
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len2 = dot(a, a);
    if len2 <= f32::EPSILON {
        return [0.0, 0.0, 0.0];
    }

    let inv_len = len2.sqrt().recip();
    [a[0] * inv_len, a[1] * inv_len, a[2] * inv_len]
}