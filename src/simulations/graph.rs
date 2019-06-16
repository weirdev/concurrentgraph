use std::io;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array, Array2, Axis};
use rand::prelude::*;
use rand::distributions::{Poisson, Gamma};

#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
pub enum AgentStatus {
    Asymptomatic,
    //Symptomatic,
    //Quarintined(usize), //time left
    Dead // TODO: This can be replaced with an Asymptomatic node that is 100% immune
}

#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
pub enum InfectionStatus {
    Infected(usize), // time left
    NotInfected(f32) // immunity
}

#[derive(Clone)]
#[derive(Debug)]
pub struct Node {
    pub status: AgentStatus,
    pub infections: Vec<InfectionStatus>
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub weights: Matrix
}

pub enum MatMulFunction {
    SingleThreaded,
    MultiThreaded,
    GPU
}

pub enum Matrix {
    Dense(Mutex<Arc<Array2<f32>>>),
    Sparse(Mutex<Arc<Vec<f32>>>)
}

pub struct Disease<'a> {
    pub name: &'a str,
    pub transmission_rate: f32,
    pub mortality_rate: f32,
    pub infection_length: usize,
    pub post_infection_immunity: f32,
    pub shedding_fun: Box<Fn(isize) -> f32>
}

fn remove_weight_mat_self_refs(weight_mat: &mut Array2<f32>) {
    for e in weight_mat.diag_mut() {
        *e = 0.0;
    }
}

pub fn generic_mat_vec_mult_multi_thread<'a, D>(mat: Arc<Array2<D>>, vector: Vec<D>, 
        op1: Arc<(Fn(&D, &D) -> D) + Sync + Send>, op2: Arc<(Fn(&D, &D) -> D) + Sync + Send>, initial_val: D) 
            -> Result<Vec<D>, &'a str> 
        where D: Copy + Sync + Send + 'static {
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
        
    } else {
        let mut result: Vec<D> = Vec::with_capacity(mat.shape()[0]);
        for _ in 0..mat.shape()[0] {
            result.push(initial_val);
        }
        let locked_result = Arc::new(Mutex::new(result));
        let arc_vector = Arc::new(vector);
        
        let threads;
        if mat.shape()[0] >= 64 {
            threads = 64;
        } else {
            threads = 1;
        }
        let mut group_size = mat.shape()[0] / threads;
        if mat.shape()[0] % threads != 0 {
            group_size += 1;
        }

        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();
        for g in 0..threads {
            let mat_ref = Arc::clone(&mat);
            let vec_ref = Arc::clone(&arc_vector);
            let op1_ref = Arc::clone(&op1);
            let op2_ref = Arc::clone(&op2);
            let res_ref = Arc::clone(&locked_result);
            let handle = thread::spawn(move || {
                let mut group_res: Vec<D> = Vec::with_capacity(group_size);
                for i in (g*group_size)..(((g+1)*group_size).min(mat_ref.shape()[0])) {
                    let mut r = initial_val;
                    for j in 0..vec_ref.len() {
                        r = op2_ref(&r, &op1_ref(&mat_ref[(i, j)], &vec_ref[j]));
                    }
                    group_res.push(r);
                }
                let mut res_lock = res_ref.lock().unwrap();
                for (i, v) in ((g*group_size)..(((g+1)*group_size).min(mat_ref.shape()[0]))).zip(group_res.iter()) {
                    res_lock[i] = *v;
                }
            });
            handles.push(handle);
        }
        while handles.len() > 0 {
            handles.pop().expect("Failed to pop thread handle").join().expect("Failed to rejoin thread");
        }
        return Ok(locked_result.lock().unwrap().to_vec());
    }
}

fn generic_mat_vec_mult_single_thread<'a, D>(mat_lock: &Mutex<Arc<Array2<D>>>, vector: Vec<D>, 
        op1: Arc<(Fn(&D, &D) -> D) + Sync + Send>, op2: Arc<(Fn(&D, &D) -> D) + Sync + Send>, initial_val: D) 
            -> Result<Vec<D>, &'a str> 
        where D: Copy + Sync + Send + 'static {
    let mat = mat_lock.lock().unwrap();
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<D> = Vec::with_capacity(mat.shape()[0]);
        
        for (i, row) in mat.outer_iter().enumerate() {
            result.push(initial_val);
            for j in 0..vector.len() {
                result[i] = op2(&result[i], &op1(&row[j], &vector[j]));
            }
        }
        return Ok(result);
    }
}

pub fn event_prob_to_timespan(p: f32, rng: &mut ThreadRng) -> isize {
    if p > 0.0 {
        let y = rng.sample(Gamma::new(1.0, ((1.0 - p) / p) as f64));
        rng.sample(Poisson::new(y)) as isize}
    else {-1}
}

// weights elements = probability of meeting at a given timestep
// output elements = number of timesteps until row meets column
// Sampling negative binomial distribution in same manner as random_negative_binomial() in
// https://github.com/numpy/numpy/blob/master/numpy/random/src/distributions/distributions.c
pub fn deterministic_weights(weights: &Array2<f32>) -> Array2<isize> {
    let mut rng = thread_rng();
    let mut determ_weights = Array2::zeros((weights.shape()[0], weights.shape()[1]));
    for (idx, p) in weights.indexed_iter() {
        determ_weights[idx] = event_prob_to_timespan(*p, &mut rng);
    }
    determ_weights
}

impl Graph {
    pub fn new_sim_graph(n: usize, connectivity: f32, infection: &Disease, bfs: bool)  -> Graph {
        let mut nodes: Vec<Node> = Vec::new();
        nodes.resize(n-1, Node {
            status: AgentStatus::Asymptomatic,
            infections: vec![InfectionStatus::NotInfected(0.1)]
        });
        
        nodes.push(Node {
            status: AgentStatus::Asymptomatic,
            infections: if bfs {vec![InfectionStatus::Infected(0)]} else {vec![InfectionStatus::Infected(infection.infection_length)]}
        });
        Graph::new_uniform_from_nodes(nodes, connectivity)
    }

    fn new_connected(n: usize) -> Graph {
        Graph {
            //nodes: (0..n).map(|_| Node::new()).collect(),
            nodes: vec![Node {
                status: AgentStatus::Asymptomatic,
                infections: Vec::new()
            }; n],
            weights: Matrix::Dense(Mutex::new(Arc::new(Array2::from_elem((n, n), 1.0))))
        }
    }

    fn new_uniform_from_nodes(nodes: Vec<Node>, connection_weights: f32) -> Graph {
        let mut w = Array2::from_elem((nodes.len(), nodes.len()), connection_weights);
        remove_weight_mat_self_refs(&mut w);
        Graph {
            weights: Matrix::Dense(Mutex::new(Arc::new(w))),
            nodes: nodes
        }
    }

    pub fn dead_count(&self) -> usize {
        self.nodes.iter()
            .filter(|n| match n.status {
                AgentStatus::Dead => true,
                _ => false
            }).count()
    }

    pub fn infected_count(&self, disease: usize) -> usize {
        self.nodes.iter()
            .filter(|n| match n.infections[disease] {
                InfectionStatus::Infected(_) => true,
                _ => false
            }).count()
    }
}
