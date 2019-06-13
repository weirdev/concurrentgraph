use std::sync::Arc;
use std::sync::Mutex;

extern crate ndarray;

use ndarray::Array2;

#[link(name = "concurrentgraph_cuda")]
extern {
    fn negative_prob_multiply_matrix_vector_cpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
    fn negative_prob_multiply_matrix_vector_gpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
}

pub fn negative_prob_multiply_matrix_vector_cpu_safe<'a>(iters: isize, mat_lock: &Mutex<Arc<Array2<f32>>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    let mat = mat_lock.lock().unwrap();
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        unsafe {
            result.set_len(mat.shape()[0]);
            negative_prob_multiply_matrix_vector_cpu(iters, mat.as_ptr(), vector.as_ptr(), result.as_mut_ptr(), mat.shape()[0], mat.shape()[1]);
        }
        return Ok(result);
    }
}

pub fn negative_prob_multiply_matrix_vector_gpu_safe<'a>(iters: isize, mat_lock: &Mutex<Arc<Array2<f32>>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    let mat = mat_lock.lock().unwrap();
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        unsafe {
            result.set_len(mat.shape()[0]);
            negative_prob_multiply_matrix_vector_gpu(iters, mat.as_ptr(), vector.as_ptr(), result.as_mut_ptr(), mat.shape()[0], mat.shape()[1]);
        }
        return Ok(result);
    }
}