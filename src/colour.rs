use std::io::Write;
 
use crate::vec3::Vec3;
 
// Type alias
pub type Colour = Vec3;
 
pub fn write_colour(out: &mut impl Write, pixel_colour: Colour) {
    // Write the translated [0, 255] value of each colour component
    let r = (255.999 * pixel_colour.x()) as i32;
    let g = (255.999 * pixel_colour.y()) as i32;
    let b = (255.999 * pixel_colour.z()) as i32;
    writeln!(out, "{} {} {}", r, g, b).expect("writing colour");
}