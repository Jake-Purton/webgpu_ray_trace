use std::sync::Arc;

use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{self, Point3, unit_vector};

pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
    pub mat: Arc<dyn Material + Send + Sync>,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3, m: Arc<dyn Material + Send + Sync>) -> Triangle {
        Triangle {
            a,
            b,
            c,
            mat: m,
        }
    }
}

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        // Möller–Trumbore intersection algorithm
        let epsilon = 1e-8;
        let edge1 = self.b - self.a;
        let edge2 = self.c - self.a;
        let h = vec3::cross(r.direction(), edge2);
        let a = vec3::dot(edge1, h);

        if a.abs() < epsilon {
            return false; // Ray is parallel to triangle
        }

        let f = 1.0 / a;
        let s = r.origin() - self.a;
        let u = f * vec3::dot(s, h);

        if u < 0.0 || u > 1.0 {
            return false;
        }

        let q = vec3::cross(s, edge1);
        let v = f * vec3::dot(r.direction(), q);

        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t = f * vec3::dot(edge2, q);

        if t < t_min || t > t_max {
            return false;
        }

        rec.t = t;
        rec.p = r.at(t);
        let outward_normal = unit_vector(vec3::cross(edge1, edge2));
        rec.set_face_normal(r, outward_normal);
        rec.mat = Some(self.mat.clone());
        true
    }
}