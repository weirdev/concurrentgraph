use std::io;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::f32::EPSILON;

use rand::prelude::*;

#[macro_use(array)]
extern crate ndarray;

extern crate num_traits;

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

fn test_sparse_stochastic(community_count: usize, community_size: usize, inter_comm_connection_prob: f32, 
                            steps: usize, disease: &Disease, mat_mul_fun: MatMulFunction) -> io::Result<()> {
    let start_time = SystemTime::now();
    let community: Vec<Node> = (0..community_size).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..community_count).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, inter_comm_connection_prob, 0.1);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    match &graph.weights {
        Matrix::Sparse(mat) => println!("Generated {} val sparse graph in {} secs", mat.lock().unwrap().values.len(), runtime.as_secs()),
        _ => ()
    }
    

    graph.nodes[0].infections = vec![InfectionStatus::Infected(disease.infection_length)];
    let start_time = SystemTime::now();
    //graph.simulate_basic_looped_stochastic(200, &[disease]);
    simulate_basic_mat_stochastic(&mut graph, steps, &[disease], mat_mul_fun);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    println!("Ran in {} secs", runtime.as_secs());

    Ok(())
}

fn test_basic_deterministic(disease: &Disease) -> io::Result<()> {
    let community: Vec<Node> = (0..100).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..200).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, 0.1, 0.1);
    
    simulate_basic_mat_bfs_gpu(&mut graph, 200, &[disease]);
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    
    
    //let start_time = SystemTime::now();
    simulate_basic_mat_bfs_cpu(&mut graph, 200, &[disease]);
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    /*let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");*/
    //println!("CPU Ran in {} secs", runtime.as_secs());
    Ok(())
}

fn random_mat_mul(iters: usize, graph_size: usize, disease: &Disease, mat_mul_fun: MatMulFunction, sparse: bool) -> io::Result<Vec<f32>> {
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
    test_mat_mul(iters, &graph.weights, vector, mat_mul_fun, 1)
}

fn test_mat_mul(iters: usize, matrix: &Matrix<f32>, vector: Vec<f32>, mat_mul_fun: MatMulFunction, gpu_restriction_factor: usize) -> io::Result<Vec<f32>> {
    let start_time = SystemTime::now();
    
    let result = match matrix {
        Matrix::Dense(m) => {
            let mat = m.lock().unwrap().clone();
            match mat_mul_fun {
                MatMulFunction::MultiThreaded => generic_mat_vec_mult_multi_thread(mat, vector.clone(), Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).expect("Run failed"), // TODO: this doesnt support computation iteration
                MatMulFunction::SingleThreaded => negative_prob_multiply_dense_matrix_vector_cpu_safe(iters as isize, mat, vector.clone()).expect("Run failed"),
                MatMulFunction::GPU => negative_prob_multiply_dense_matrix_vector_gpu_safe(iters as isize, mat, vector.clone()).expect("Run failed")
            }
        },
        Matrix::Sparse(sm) => {
            let sp_mat = sm.lock().unwrap().clone();
            match mat_mul_fun {
                MatMulFunction::GPU => {
                    let gpu_alloc = npmmv_csr_gpu_allocate_safe(sp_mat.rows, sp_mat.columns, sp_mat.values.len());
                    npmmv_gpu_set_csr_matrix_safe(sp_mat.get_ptrs(), gpu_alloc, sp_mat.rows, sp_mat.values.len());
                    npmmv_gpu_set_in_vector_safe(vector, NpmmvAllocations::Sparse(gpu_alloc));
                    for _ in 0..iters {
                        npmmv_csr_gpu_compute_safe(gpu_alloc, sp_mat.rows, gpu_restriction_factor);
                    }
                    let res = npmmv_gpu_get_out_vector_safe(NpmmvAllocations::Sparse(gpu_alloc), sp_mat.rows);
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
    let graph_size = 10_000;
    let mut graph = Graph::new_sim_graph(graph_size, 0.3, disease, false);

    //let vector: Vec<f32> = (0..graph_size).map(|_| random::<f32>()).collect();
    let mut vector: Vec<f32> = (0..graph_size).map(|_| 0.0).collect();
    vector[40] = 0.9;
    let dense_result = test_mat_mul(1, &graph.weights, vector.clone(), MatMulFunction::GPU, 1)?;

    let sp_mat;
    {
        let w = match &graph.weights {
            Matrix::Dense(ref m) => m,
            _ => panic!("Graph must be dense here")
        }.lock().unwrap();

        sp_mat = Matrix::Sparse(Mutex::new(Arc::new(CsrMatrix::from_dense(w.clone()))));
    }
    graph.weights = sp_mat;

    let sparse_result = test_mat_mul(1, &graph.weights, vector.clone(), MatMulFunction::GPU, 1)?;

    for i in 0..dense_result.len() {
        if dense_result[i] - sparse_result[i] > EPSILON*100.0 {
            panic!("Result mismatch at {}, dense: {} sparse: {}", i, dense_result[i], sparse_result[i]);
        }
    }
    println!("item 39 in result: {}", dense_result[39]);
    println!("item 40 in result: {}", dense_result[40]);
    Ok(())
}

fn mat_mul_test3(disease: &Disease, size: usize, iters: usize, gpu_restriction_factor: usize, sparsity: f32) -> io::Result<()> {
    let mat = Matrix::Sparse(Mutex::new(Arc::new(CsrMatrix::new_with_conn_prob(size, size, sparsity))));
    let vector: Vec<f32> = (0..size).map(|_| random::<f32>()).collect();
    test_mat_mul(iters, &mat, vector.clone(), MatMulFunction::GPU, gpu_restriction_factor)?;

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
    /*
    println!("Sparsity factor 0.001");
    println!("10_000 nodes, 1_000 steps");
    test_sparse_stochastic(10_000, 1, 0.001, 1_000, &flu, MatMulFunction::GPU)?;
    println!("30_000 nodes, 100 steps");
    test_sparse_stochastic(30_000, 1, 0.001, 100, &flu, MatMulFunction::GPU)?;
    println!("100_000 nodes, 10 steps");
    test_sparse_stochastic(100_000, 1, 0.001, 10, &flu, MatMulFunction::GPU)?;
    */
    
    test_basic_deterministic(&flu)?;

    //test_basic_deterministic(&flu)?;
    //mat_mul_test1(&flu)?;
    //mat_mul_test2(&flu)?;

    //mat_mul_test3(&flu, 100_000, 3, 4, 0.01)?;

    Ok(())
}
