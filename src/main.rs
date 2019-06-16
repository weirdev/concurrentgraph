use std::io;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::prelude::*;

#[macro_use(array)]
extern crate ndarray;

mod graph;

use graph::*;
use external_adaptor::*;

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

fn test_mat_mul(iters: isize, graph_size: usize, disease: &Disease, mat_mul_fun: MatMulFunction) -> io::Result<()> {
    let graph = Graph::new_sim_graph(graph_size, 0.3, disease, false);
    let vector: Vec<f32> = (0..graph_size).map(|_| random::<f32>()).collect();
    let start_time = SystemTime::now();
    
    match mat_mul_fun {
        MatMulFunction::MultiThreaded => generic_mat_vec_mult_multi_thread(&graph.weights, vector.clone(), Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0),
        MatMulFunction::SingleThreaded => negative_prob_multiply_matrix_vector_cpu_safe(iters, &graph.weights, vector.clone()),
        MatMulFunction::GPU => negative_prob_multiply_matrix_vector_gpu_safe(iters, &graph.weights, vector.clone())
    }.expect("Run failed");

    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("Ran in {} secs", runtime.as_secs());
    Ok(())
}

fn mat_mul_test1(disease: &Disease) -> io::Result<()> {
    println!("GPU test");
    println!("100 iters, 10_000 size mat");
    test_mat_mul(100, 10_000, &disease, MatMulFunction::GPU)?;
    println!("100 iters, 15_000 size mat");
    test_mat_mul(100, 15_000, &disease, MatMulFunction::GPU)?;

    println!("CPU single thread test");
    println!("100 iters, 10_000 size mat");
    test_mat_mul(100, 10_000, &disease, MatMulFunction::SingleThreaded)?;
    println!("100 iters, 15_000 size mat");
    test_mat_mul(100, 15_000, &disease, MatMulFunction::SingleThreaded)?;
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
    test_basic_deterministic(&flu)?;
    //mat_mul_test1(&flu)?;

    Ok(())
}
