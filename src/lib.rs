#[cfg(target_arch = "wasm32")]
use std::sync::OnceLock;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod read_obj;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn default_width() -> u32 {
    320
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn default_height() -> u32 {
    180
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn scene_bytes() -> Vec<u8> {
    scene_data().clone()
}

#[cfg(target_arch = "wasm32")]
fn scene_data() -> &'static Vec<u8> {
    static SCENE: OnceLock<Vec<u8>> = OnceLock::new();
    SCENE.get_or_init(|| read_obj::read_obj_vertices_from_bytes(include_bytes!("../suzanne.obj")))
}
