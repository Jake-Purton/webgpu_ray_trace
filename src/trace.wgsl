struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

struct Material {
    // colour of the surface when it reflects light
    emmission: vec4<f32>, // first 3 f32s is the emmission colour, last is the coefficient (emmission strength)
    albedo: vec4<f32>,
    material_type: u32, // metallic or lambertian
}

struct Camera {
    origin: vec3<f32>,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
}

@group(0) @binding(0)
var<storage, read> input: array<Triangle>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@group(0) @binding(3)
var<storage, read> materials: array<Material>;

struct Params {
    width: u32,
    height: u32,
    camera: Camera,
    depth: u32,
    samples: u32,
    pad3: u32,
    pad4: u32,
};

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    face_normal: vec3<f32>,
    color: vec3<f32>,
    did_hit: bool,
    emmitted_colour: vec3<f32>,
    emmitted_strength: f32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

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
    let fy = f32(params.height - 1u - y) / f32(params.height - 1u);

    let ray = get_ray(params.camera, fx, fy);

    var pixel_color = vec3(0.0, 0.0, 0.0);

    for (var s: u32 = 0; s<params.samples; s++) {
        let seed = make_seed(x, y, s, 0u);

        let fx = (f32(x) + random_double(seed)) / f32(params.width - 1u);
        let fy = (f32(params.height - 1u - y) + random_double(seed + 1)) / f32(params.height - 1u);
        let ray = get_ray(params.camera, fx, fy);
        pixel_color += ray_color_iter(ray, params.depth, seed);
    }

    let scale = 1.0 / f32(params.samples);

    let c = pixel_color * scale;

    let r: u32 = u32(c.x * 255.0);
    let g: u32 = u32(c.y * 255.0);
    let b: u32 = u32(c.z * 255.0);
    

    output[index] = (r << 16u) | (g << 8u) | b;

    // output[index] = 
}

fn hit_triangle (r: Ray, t_min: f32, triangle: Triangle) -> HitRecord {

    var hr: HitRecord = HitRecord(
        0.0,
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false,
        vec3<f32>(0.0, 0.0, 0.0),
        0.0
    );

    let epsilon = 1e-8;
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let h = cross(r.direction, edge2.xyz);
    let a = dot(edge1.xyz, h);

    if abs(a) < epsilon {
        return hr;
    }

    let f = 1.0 / a;
    let s = r.origin - triangle.a.xyz;
    let u = f * dot(s, h);

    if u < 0.0 || u > 1.0 {
        return hr;
    }

    let q = cross(s, edge1.xyz);
    let v = f * dot(r.direction, q);

    if v < 0.0 || u + v > 1.0 {
        return hr;
    }

    let t = f * dot(edge2.xyz, q);

    // if t < t_min || t > t_max {
    if t < t_min {
        return hr;
    }

    hr.t = t;
    hr.point = (r.direction * t) + r.origin;

    let outward_normal = normalize(cross(edge1.xyz, edge2.xyz));

    if dot(r.direction, outward_normal) < 0.0 {
        hr.face_normal = outward_normal;
    } else {
        hr.face_normal = -outward_normal;
    };

    hr.did_hit = true;

    let material = materials[u32(triangle.a.w)];

    hr.color = material.albedo.xyz;
    hr.emmitted_colour = material.emmission.xyz;
    hr.emmitted_strength = material.emmission.w;

    return hr;
}

fn ray_color_iter(r_in: Ray, max_depth: u32, seed: u32) -> vec3<f32> {
    var incoming_light = vec3<f32>(0.0, 0.0, 0.0);
    var color = vec3<f32>(1.0, 1.0, 1.0); // accumulated color
    var ray = r_in;
    var depth = max_depth;

    loop {
        if depth == 0u {
            break;
        }

        let hr = calculate_collisions(ray);

        if hr.did_hit {
            ray.origin = hr.point;
            ray.direction = random_hemisphere_direction(hr.face_normal, seed+depth);
            let emitted_light = hr.emmitted_colour * hr.emmitted_strength;
            incoming_light += emitted_light * color;
            color *= hr.color;
        } else {
            // background emitted color * emmission_strength
            let e = vec3(0.99, 0.99, 0.99) * 0.0;
            incoming_light += e * color;
            break;
        }

        depth = depth - 1u;
    }

    return incoming_light;
}

fn calculate_collisions(ray: Ray) -> HitRecord {
    var hr: HitRecord = HitRecord(
        1e30, // large initial t
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false,
        vec3<f32>(0.0, 0.0, 0.0),
        0.0
    );

    // Find nearest hit
    for (var i = 0u; i < arrayLength(&input); i = i + 1u) {
        let triangle = input[i];
        let hit2 = hit_triangle(ray, 0.001, triangle);

        if (!hr.did_hit) || (hit2.t < hr.t && hit2.did_hit){
            hr = hit2;
        }
    }

    return hr;
}

fn random_hemisphere_direction(normal: vec3<f32>, seed: u32) -> vec3<f32> {
    let z = random_double(seed) * 2.0 - 1.0;
    let a = random_double(seed + 1u) * 6.28318530718;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let v = vec3<f32>(
        r * cos(a), 
        r * sin(a),
        z
    );

    if dot(v, normal) < 0 {
        return -v;
    } else {
        return v;
    }
}

fn random_double(seed:u32) -> f32 {

    let a = hash_u32(seed);

    let fa = f32(a);

    return (fa)* (1.0 / 4294967296.0);
}

fn get_ray(c: Camera, u: f32, v: f32) -> Ray {
    return Ray(
        c.origin,
        c.lower_left_corner + u * c.horizontal + v * c.vertical - c.origin,
    );
}


fn hash_u32(seed: u32) -> u32 {
    var v = seed;
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    v = (v >> 22u) ^ v;
    return v;
}

fn make_seed(x: u32, y: u32, sample: u32, bounce: u32) -> u32 {
    var s = x * 1973u + y * 9277u + sample * 26699u + bounce * 3181u;
    return hash_u32(s);
}
