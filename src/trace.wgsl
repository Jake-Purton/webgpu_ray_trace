struct Triangle {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
}

@group(0) @binding(0)
var<storage, read> input: array<Triangle>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
};

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    face_normal: vec3<f32>,
    did_hit: bool,
}

struct Ray {
    direction: vec3<f32>,
    origin: vec3<f32>
}

@group(0) @binding(2)
var<uniform> params: Params;

// learnt that if you dont use the buffer in the code it wont be generate and you will get an error

// one thread per pixel!
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    let d = input[0];

    // Guard against extra threads
    if (x >= params.width || y >= params.height) {
        return;
    }

    // 1D index into output buffer
    let index = y * params.width + x;

    // Normalized coordinates [0, 1]
    let fx = f32(x) / f32(params.width - 1u);
    let fy = f32(y) / f32(params.height - 1u);

    // Simple gradient
    let r: u32 = u32(fx * 255.0);
    let g: u32 = u32(fy * 255.0);
    let b: u32 = 128u;

    // Pack as 0x00RRGGBB
    output[index] = (r << 16u) | (g << 8u) | b;
}

fn hit_triangle (r: Ray, t_min: f32, triangle: Triangle) -> HitRecord {

    var hr: HitRecord = HitRecord(
        0.0,
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false
    );

    let epsilon = 1e-8;
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let h = cross(r.direction, edge2);
    let a = dot(edge1, h);

    if abs(a) < epsilon {
        return hr;
    }

    let f = 1.0 / a;
    let s = r.origin - triangle.a;
    let u = f * dot(s, h);

    if u < 0.0 || u > 1.0 {
        return hr;
    }

    let q = cross(s, edge1);
    let v = f * dot(r.direction, q);

    if v < 0.0 || u + v > 1.0 {
        return hr;
    }

    let t = f * dot(edge2, q);

    // if t < t_min || t > t_max {
    if t < t_min {
        return hr;
    }

    hr.t = t;
    hr.point = (r.direction * t) + r.origin;

    let outward_normal = normalize(cross(edge1, edge2));

    if dot(r.direction, outward_normal) < 0.0 {
        hr.face_normal = outward_normal;
    } else {
        hr.face_normal = -outward_normal;
    };

    hr.did_hit = true;

    return hr;
}

fn ray_color(r: Ray, depth: u32) -> vec3<f32> {
    // If we've exceeded the ray bounce limit, no more light is gathered

    if depth <= 0 {
        return vec3(0.0, 0.0, 0.0);
    }

    var hr: HitRecord = HitRecord(
        0.0,
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false
    );

    for (var i = 0u; i < arrayLength(&input); i = i + 1u) {

        let triangle = input[i];

        let hit2 = hit_triangle(r, 0.001, triangle);

        if hit2.t < hr.t {
            hr = hit2;
        }

    }

    if hr.did_hit {

        let scattered_ray = scatter(hr);
        let attenuation = vec3(0.8, 0.8, 0.0);

        return attenuation * ray_color(scattered_ray, depth - 1);

    }

    let unit_direction = normalize(r.direction);
    let t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);

}

fn scatter(
    rec: HitRecord,
) -> Ray {

    let normal = normalize(rec.face_normal);
    var scatter_direction = normal + random_unit_vector();
    if scatter_direction.x < 1.0e-8 && scatter_direction.y < 1.0e-8 && scatter_direction.z < 1.0e-8 {
        scatter_direction = normal;
    }
    return Ray(rec.point, scatter_direction);
}

fn random_unit_vector() -> vec3<f32> {
    return vec3(0.0, 0.0, 0.0);
}

