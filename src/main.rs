use std::io;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::f32::EPSILON;

use rand::prelude::*;

#[macro_use(array)]
extern crate ndarray;

extern crate concurrentgraph_cuda_sys;

use concurrentgraph_cuda_sys::*;

mod simulations;

use simulations::*;

fn test_basic_stochastic(disease: &Disease, mat_mul_fun: MatMulFunction) -> io::Result<()> {
    let mut graph = Graph::new_sim_graph(10_000, 0.3, disease, false);
    let start_time = SystemTime::now();
    //graph.simulate_basic_looped_stochastic(200, &[disease]);
    simulate_basic_mat_stochastic(&mut graph, 200, &[disease], mat_mul_fun);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    println!("Ran in {} secs", runtime.as_secs());
    Ok(())
}

fn test_sparse_stochastic(disease: &Disease, mat_mul_fun: MatMulFunction) -> io::Result<()> {
    let start_time = SystemTime::now();
    let community: Vec<Node> = (0..100).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..1000).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, 0.001, 0.1);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    match &graph.weights {
        Matrix::Sparse(mat) => println!("Generated {} val sparse graph in {} secs", mat.lock().unwrap().values.len(), runtime.as_secs()),
        _ => ()
    }
    

    graph.nodes[0].infections = vec![InfectionStatus::Infected(disease.infection_length)];
    let start_time = SystemTime::now();
    //graph.simulate_basic_looped_stochastic(200, &[disease]);
    simulate_basic_mat_stochastic(&mut graph, 200, &[disease], mat_mul_fun);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    println!("Ran in {} secs", runtime.as_secs());

    Ok(())
}

fn test_basic_deterministic(disease: &Disease) -> io::Result<()> {
    let mut graph = Graph::new_sim_graph(100, 0.3, disease, true);
    let start_time = SystemTime::now();
    simulate_looped_bfs(&mut graph, 200, &[disease]);
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    println!("Ran in {} secs", runtime.as_secs());
    
    let mut graph = Graph::new_sim_graph(100, 0.3, disease, false);
    let start_time = SystemTime::now();
    simulate_simplistic_mat_deterministic(&mut graph, 200, &[disease]);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    println!("Ran in {} secs", runtime.as_secs());
    Ok(())
}

fn random_mat_mul(iters: isize, graph_size: usize, disease: &Disease, mat_mul_fun: MatMulFunction, sparse: bool) -> io::Result<Vec<f32>> {
    let mut graph = Graph::new_sim_graph(graph_size, 0.3, disease, false);
    let mat;
    {
        let w = match &graph.weights {
            Matrix::Dense(ref m) => m,
            _ => panic!("Graph must be dense here")
        };
        if sparse {
            mat = Matrix::Sparse(Mutex::new(Arc::new(CsrMatrix::from_dense(w.lock().unwrap().clone()))));
        } else {
            mat = graph.weights;
        }
    }
    graph.weights = mat;
    let vector: Vec<f32> = (0..graph_size).map(|_| random::<f32>()).collect();
    test_mat_mul(iters, &graph, vector, mat_mul_fun)
}

fn test_mat_mul(iters: isize, graph: &Graph, vector: Vec<f32>, mat_mul_fun: MatMulFunction) -> io::Result<Vec<f32>> {
    let start_time = SystemTime::now();
    
    let result = match &graph.weights {
        Matrix::Dense(m) => {
            let mat = m.lock().unwrap().clone();
            match mat_mul_fun {
                MatMulFunction::MultiThreaded => generic_mat_vec_mult_multi_thread(mat, vector.clone(), Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).expect("Run failed"), // TODO: this doesnt support computation iteration
                MatMulFunction::SingleThreaded => negative_prob_multiply_dense_matrix_vector_cpu_safe(iters, mat, vector.clone()).expect("Run failed"),
                MatMulFunction::GPU => negative_prob_multiply_dense_matrix_vector_gpu_safe(iters, mat, vector.clone()).expect("Run failed")
            }
        },
        Matrix::Sparse(sm) => {
            let sp_mat = sm.lock().unwrap().clone();
            match mat_mul_fun {
                MatMulFunction::GPU => {
                    let gpu_alloc = npmmv_csr_gpu_allocate_safe(sp_mat.rows, sp_mat.columns, sp_mat.values.len());
                    npmmv_gpu_set_csr_matrix_safe(sp_mat.get_ptrs(), gpu_alloc, sp_mat.rows, sp_mat.values.len());
                    npmmv_gpu_set_in_vector_safe(vector, GpuAllocations::Sparse(gpu_alloc));
                    for _ in 0..iters {
                        npmmv_csr_gpu_compute_safe(gpu_alloc, sp_mat.rows);
                    }
                    let res = npmmv_gpu_get_out_vector_safe(GpuAllocations::Sparse(gpu_alloc), sp_mat.rows);
                    npmmv_csr_gpu_free_safe(gpu_alloc);
                    res
                },
                _ => panic!("Sparse matrices not implemented on CPU yet")
            }
        }
    };

    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("Ran in {} secs", runtime.as_secs());
    Ok(result)
}

fn mat_mul_test1(disease: &Disease) -> io::Result<()> {
    println!("GPU test");
    println!("100 iters, 10_000 size mat");
    random_mat_mul(100, 10_000, &disease, MatMulFunction::GPU, false)?;
    println!("100 iters, 15_000 size mat");
    random_mat_mul(100, 15_000, &disease, MatMulFunction::GPU, false)?;

    println!("CPU single thread test");
    println!("100 iters, 10_000 size mat");
    random_mat_mul(100, 10_000, &disease, MatMulFunction::SingleThreaded, false)?;
    println!("100 iters, 15_000 size mat");
    random_mat_mul(100, 15_000, &disease, MatMulFunction::SingleThreaded, false)?;
    Ok(())
}

/// Verify that sparse multiplication and dense multiplication reach the same result
fn mat_mul_test2(disease: &Disease) -> io::Result<()> {
    let mut graph = Graph::new_sim_graph(100, 0.3, disease, false);

    let vector: Vec<f32> = (0..100).map(|_| random::<f32>()).collect();
    let dense_result = test_mat_mul(1, &graph, vector.clone(), MatMulFunction::GPU)?;

    let sp_mat;
    {
        let w = match &graph.weights {
            Matrix::Dense(ref m) => m,
            _ => panic!("Graph must be dense here")
        }.lock().unwrap();

        sp_mat = Matrix::Sparse(Mutex::new(Arc::new(CsrMatrix::from_dense(w.clone()))));
    }
    graph.weights = sp_mat;

    let sparse_result = test_mat_mul(1, &graph, vector.clone(), MatMulFunction::GPU)?;

    for i in 0..dense_result.len() {
        if dense_result[i] - sparse_result[i] > EPSILON*100.0 {
            panic!("Result mismatch at {}, dense: {} sparse: {}", i, dense_result[i], sparse_result[i]);
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let flu = Disease {
        name: "flu",
        transmission_rate: 0.3,
        mortality_rate: 0.1,
        infection_length: 10,
        post_infection_immunity: 0.99,
        //shedding_fun: Box::new(|d| if d > 0 {1.0 / (12-d) as f32} else {0.0})
        shedding_fun: Box::new(|d| if d > 0 {1.0 / d as f32} else {0.0})
    };

    //test_basic_stochastic(&flu, MatMulFunction::SingleThreaded)?;
    //test_basic_stochastic(&flu, MatMulFunction::GPU)?;
    //test_sparse_stochastic(&flu, MatMulFunction::GPU)?;

    //test_basic_deterministic(&flu)?;
    //mat_mul_test1(&flu)?;
    mat_mul_test2(&flu)?;

    Ok(())
}
