pub fn get_ray(&self, u: f32, v: f32) -> Ray {
    Ray::new(
        self.origin,
        self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin,
    )
}

add all that camera stuff to a camera struct and add it to params