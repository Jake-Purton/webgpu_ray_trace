struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    face_normal: vec3<f32>,
    color: vec3<f32>,
    did_hit: bool,
}

@group(0) @binding(0)
var<storage, read> input: array<Triangle>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

fn hit_triangle (r: Ray, t_min: f32, triangle: Triangle) -> HitRecord {

    var hr: HitRecord = HitRecord(
        0.0,
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false,
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

    // choose a colour
    hr.color = vec3(0.7, 0.8, 0.9);

    return hr;
}

fn calculate_collisions(ray: Ray) -> HitRecord {
    var hr: HitRecord = HitRecord(
        1e30, // large initial t
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        false,
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

// one thread per pixel!
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    let d = input[0];

    // Guard against extra threads
    if (x >= 400 || y >= 225) {
        return;
    }

    // 1D index into output buffer
    let index = y * 400 + x;

    // Normalized coordinates [0, 1]
    let fx = f32(x) / f32(400 - 1u);
    let fy = f32(225 - 1u - y) / f32(225 - 1u);

    let ray = get_ray(fx, fy);
    
    let seed = make_seed(x, y, 0u);

    var c = ray_color_iter(ray, 4u, seed);

    let r: u32 = u32(c.x * 255.0);
    let g: u32 = u32(c.y * 255.0);
    let b: u32 = u32(c.z * 255.0);
    
    output[index] = (r << 16u) | (g << 8u) | b;
}

fn get_ray(u: f32, v: f32) -> Ray {

    let aspect_ratio = 16.0 / 9.0;

    let origin = vec3<f32>(0.0, 0.0, 0.0);
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let horizontal = vec3<f32>(viewport_width, 0.0, 0.0);
    let vertical = vec3<f32>(0.0, viewport_height, 0.0);
    let focal_length = vec3<f32>(0.0, 0.0, 1.0);
    
    let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focal_length;

    return Ray (
        origin,
        lower_left_corner + u * horizontal + v * vertical - origin,
    );
}

fn random_hemisphere_direction(normal: vec3<f32>, seed: u32) -> vec3<f32> {
    let z = random_float(seed) * 2.0 - 1.0;
    let a = random_float(seed + 1u) * 6.28318530718;
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

fn random_float(seed:u32) -> f32 {

    let a = hash_u32(seed);

    let fa = f32(a);

    return (fa)* (1.0 / 4294967296.0);
}

fn hash_u32(seed: u32) -> u32 {
    var v = seed;
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    v = (v >> 22u) ^ v;
    return v;
}

fn make_seed(x: u32, y: u32, sample: u32) -> u32 {
    var s = x * 1973u + y * 9277u + sample * 26699u;
    return hash_u32(s);
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

            // let emitted_light = hr.emmitted_colour * hr.emmitted_strength;
            // incoming_light += emitted_light * color;
            color *= hr.color;
        } else {
            // background emitted color * emmission_strength
            let e = vec3(0.88, 0.88, 0.99) * .99;
            incoming_light += e * color;
            break;
        }

        depth = depth - 1u;
    }

    return incoming_light;
}