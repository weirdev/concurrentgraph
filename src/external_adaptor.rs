use std::sync::Arc;
use std::sync::Mutex;

extern crate ndarray;

use ndarray::{Array, Array2, Axis};

#[link(name = "concurrentgraph-cuda", kind = "static")]
extern {
    fn negative_prob_multiply_matrix_vector_external(matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
}

pub fn negative_prob_multiply_matrix_vector<'a>(mat_lock: &Mutex<Arc<Array2<f32>>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    let mat = mat_lock.lock().unwrap();
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        unsafe {
            negative_prob_multiply_matrix_vector_external(mat.as_ptr(), vector.as_ptr(), result.as_mut_ptr(), mat.shape()[0], mat.shape()[1]);
        }
        return Ok(result);
    }
}