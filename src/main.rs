#![allow(dead_code)]
use std::io;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::f32::EPSILON;

use rand::prelude::*;

#[allow(unused_imports)]
#[macro_use(array)]
extern crate ndarray;

extern crate num_traits;

extern crate rayon;

extern crate concurrentgraph_cuda_sys;

use concurrentgraph_cuda_sys::*;

mod simulations;

use simulations::*;

fn test_basic_stochastic(disease: &Disease, mat_mul_fun: MatMulFunction) -> io::Result<()> {
    let mut graph = Graph::new_sim_graph(10_000, 0.3, disease, false);
    let start_time = SystemTime::now();
    //graph.simulate_basic_looped_stochastic(200, &[disease]);
    //for _ in 0..iters {
        simulate_basic_mat_stochastic(&mut graph, 200, &[disease], mat_mul_fun);
    //}
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
    println!("Ran in {} secs", runtime.as_secs());
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));

    Ok(())
}

fn test_basic_deterministic(disease: &Disease) -> io::Result<()> {
    let community: Vec<Node> = (0..100).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..200).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, 0.001, 0.1);
    
    let start_time = SystemTime::now();
    simulate_basic_mat_bfs_gpu(&mut graph, 100, &[disease]);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("total GPU Ran in {} secs", runtime.as_secs());
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    
    let community: Vec<Node> = (0..100).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..200).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, 0.001, 0.1);
    let start_time = SystemTime::now();
    simulate_basic_mat_bfs_cpu(&mut graph, 100, &[disease]);
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("total CPU Ran in {} secs", runtime.as_secs());
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
                MatMulFunction::MultiThreaded => {
                    (0..(iters-1)).for_each(|_| {generic_mat_vec_mult_multi_thread(mat.clone(), vector.clone(), Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).expect("Run failed");});
                    generic_mat_vec_mult_multi_thread(mat.clone(), vector.clone(), Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).expect("Run failed")
                }, // TODO: this doesnt support computation iteration
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

fn mat_mul_test3(size: usize, iters: usize, gpu_restriction_factor: usize, sparsity: f32) -> io::Result<()> {
    let mat = Matrix::Sparse(Mutex::new(Arc::new(CsrMatrix::new_with_conn_prob(size, size, sparsity))));
    let vector: Vec<f32> = (0..size).map(|_| random::<f32>()).collect();
    test_mat_mul(iters, &mat, vector.clone(), MatMulFunction::GPU, gpu_restriction_factor)?;

    Ok(())
}

fn mat_mul_test4(iters: usize, disease: &Disease, mat_mul_fun: MatMulFunction) {
    println!("{} iters, 10_000 size mat", iters);
    random_mat_mul(iters, 10_000, &disease, mat_mul_fun, false).unwrap();
}

fn test_hospital_graph_mat_mul(file: &str, iters: usize) {
    let sp_mat = CsrMatrix::read_from_adj_list_file(file);
    println!("loaded {}", file);
    let mut vector: Vec<f32> = (0..sp_mat.rows).map(|_| 0.0).collect();
    vector[40] = 0.9;

    let sp_mat = sp_mat.randomize_rows();

    let gpu_alloc = npmmv_csr_gpu_allocate_safe(sp_mat.rows, sp_mat.columns, sp_mat.values.len());
    npmmv_gpu_set_csr_matrix_safe(sp_mat.get_ptrs(), gpu_alloc, sp_mat.rows, sp_mat.values.len());

    let start_time = SystemTime::now();
    npmmv_gpu_set_in_vector_safe(vector.clone(), NpmmvAllocations::Sparse(gpu_alloc));
    for _ in 0..iters {
        npmmv_csr_gpu_compute_safe(gpu_alloc, sp_mat.rows, 1);
    }
    let _res = npmmv_gpu_get_out_vector_safe(NpmmvAllocations::Sparse(gpu_alloc), sp_mat.rows);
    npmmv_csr_gpu_free_safe(gpu_alloc);
    
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("Ran nonordered rows in {} secs", runtime.as_secs());


    let sp_mat = sp_mat.sort_rows();

    let gpu_alloc = npmmv_csr_gpu_allocate_safe(sp_mat.rows, sp_mat.columns, sp_mat.values.len());
    npmmv_gpu_set_csr_matrix_safe(sp_mat.get_ptrs(), gpu_alloc, sp_mat.rows, sp_mat.values.len());

    let start_time = SystemTime::now();
    npmmv_gpu_set_in_vector_safe(vector.clone(), NpmmvAllocations::Sparse(gpu_alloc));
    for _ in 0..iters {
        npmmv_csr_gpu_compute_safe(gpu_alloc, sp_mat.rows, 1);
    }
    let _res = npmmv_gpu_get_out_vector_safe(NpmmvAllocations::Sparse(gpu_alloc), sp_mat.rows);
    npmmv_csr_gpu_free_safe(gpu_alloc);
    
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("Ran sorted rows in {} secs", runtime.as_secs());
}

fn time_sssp(file: &str, iters: usize) {
    let sp_mat = CsrMatrix::new_with_conn_prob(100000, 100000, 0.1); //CsrMatrix::read_from_adj_list_file(file);
    println!("loaded {}", file);
    
    let start_time = SystemTime::now();
    for _ in 0..iters {
        sssp_safe(sp_mat.get_ptrs(), sp_mat.rows, sp_mat.values.len());
    }
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("Ran sssp in {} secs {} iters", runtime.as_secs(), iters);
}

fn compare_stochastic_deterministic(disease: &Disease, iters: usize) -> io::Result<()> {
    let community: Vec<Node> = (0..100).map(|_| Node { status: AgentStatus::Asymptomatic, infections: vec![InfectionStatus::NotInfected(0.1)] }).collect();
    let communities: Vec<Vec<Node>> = (0..100).map(|_| community.clone()).collect();
    let mut graph = Graph::new_sparse_from_communities(communities, 0.2, 0.01, 0.1);

    let steps = 200;
    /*
    let start_time = SystemTime::now();
    simulate_basic_mat_stochastic(&mut graph, steps, &[disease], MatMulFunction::SingleThreaded);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("total CPU st stoch Ran in {} secs for {} steps", runtime.as_secs(), steps);
    */

    let start_time = SystemTime::now();
    for _ in 0..iters {
        simulate_basic_mat_stochastic(&mut graph, steps, &[disease], MatMulFunction::GPU);
    }
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("total GPU stoch Ran in {} secs for {} steps", runtime.as_secs(), steps);

    let start_time = SystemTime::now();
    for _ in 0..iters {
        simulate_basic_mat_bfs_cpu(&graph, steps, &[disease]);
    }
    //graph.simulate_basic_looped_deterministic_shedding_incorrect(200, &[disease]);
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("total CPU determ Ran in {} secs", runtime.as_secs());
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
    //test_basic_stochastic(&flu, MatMulFunction::MultiThreaded)?;
    mat_mul_test4(5, &flu, MatMulFunction::MultiThreaded);
    mat_mul_test4(5, &flu, MatMulFunction::SingleThreaded);
    /*
    println!("Sparsity factor 0.001");
    println!("10_000 nodes, 1_000 steps");
    */
    //test_sparse_stochastic(10_000, 1, 0.1, 2000, &flu, MatMulFunction::GPU)?;
    /*
    println!("30_000 nodes, 100 steps");
    test_sparse_stochastic(30_000, 1, 0.001, 100, &flu, MatMulFunction::GPU)?;
    println!("100_000 nodes, 10 steps");
    test_sparse_stochastic(100_000, 1, 0.001, 10, &flu, MatMulFunction::GPU)?;
    */
    
    //test_basic_deterministic(&flu)?;

    //test_basic_deterministic(&flu)?;
    //mat_mul_test1(&flu)?;
    //mat_mul_test2(&flu)?;

    //mat_mul_test3(10_000, 20, 1, 0.01)?;

    //test_hospital_graph_mat_mul("obsSparse5.adjlist", 10000);
    //test_hospital_graph_mat_mul("obsMod5.adjlist", 5000);
    //test_hospital_graph_mat_mul("obsDense5.adjlist", 1000);

    //compare_stochastic_deterministic(&flu, 1);

    //time_sssp("obsSparse5.adjlist", 1000);

    Ok(())
}
