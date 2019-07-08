use std::io;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::time::{SystemTime, UNIX_EPOCH};
use num_traits::identities::Zero;

use ndarray::{Array, Array2, Axis};
use rand::prelude::*;
use rand::distributions::{Poisson, Gamma};

use concurrentgraph_cuda_sys::{CsrFloatMatrixPtrs, CsrIntMatrixPtrs};

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
    pub weights: Matrix<f32>
}

pub enum MatMulFunction {
    SingleThreaded,
    MultiThreaded,
    GPU
}

// Compressed sparse row matrix
#[derive(Clone)]
pub struct CsrMatrix<T> {
    pub rows: usize,
    pub columns: usize,
    pub cum_row_indexes: Vec<usize>,
    pub column_indexes: Vec<usize>,
    pub values: Vec<T>
}

impl<T> CsrMatrix<T> 
        where T: Clone + Copy + Zero + PartialOrd {
    pub fn new(rows: usize, columns: usize) -> CsrMatrix<T> {
        CsrMatrix {
            rows: rows,
            columns: columns,
            cum_row_indexes: Vec::new(),
            column_indexes: Vec::new(),
            values: Vec::new()
        }
    }

    pub fn from_dense(mat: Arc<Array2<T>>) -> CsrMatrix<T> {
        let mut sparse_mat = CsrMatrix::new(mat.shape()[0], mat.shape()[1]);
        for i in 0..mat.shape()[0] {
            sparse_mat.cum_row_indexes.push(sparse_mat.values.len());
            for j in 0..mat.shape()[1] {
                if mat[(i, j)] <= T::zero() {
                    sparse_mat.column_indexes.push(j);
                    sparse_mat.values.push(mat[(i, j)])
                }
            }
        }
        sparse_mat.cum_row_indexes.push(sparse_mat.values.len());

        sparse_mat
    }

    pub fn to_dense(&self) -> Array2<T> {
        let mut mat = Array2::zeros((self.rows, self.columns));
        for i in 0..(self.cum_row_indexes.len() - 1) {
            for j in self.cum_row_indexes[i]..self.cum_row_indexes[i+1] {
                mat[(i, self.column_indexes[j])] = self.values[j];
            }
        }
        mat
    }
}

impl CsrMatrix<f32> {
    pub fn new_with_conn_prob(rows: usize, columns: usize, conn_prob: f32) -> CsrMatrix<f32> {
        let mut sp_mat = CsrMatrix::new(rows, columns);
        for i in 0..rows {
            sp_mat.cum_row_indexes.push(sp_mat.values.len());
            for j in 0..columns {
                if random::<f32>() < conn_prob {
                    sp_mat.column_indexes.push(j);
                    sp_mat.values.push(0.5);
                }
            }
        }
        sp_mat.cum_row_indexes.push(sp_mat.values.len());
        sp_mat
    }

    pub fn get_ptrs(&self) -> CsrFloatMatrixPtrs {
        CsrFloatMatrixPtrs {
            cum_row_indexes: self.cum_row_indexes.as_ptr(),
            column_indexes: self.column_indexes.as_ptr(),
            values: self.values.as_ptr()
        }
    }
}

impl CsrMatrix<isize> {
    pub fn get_ptrs(&self) -> CsrIntMatrixPtrs {
        CsrIntMatrixPtrs {
            cum_row_indexes: self.cum_row_indexes.as_ptr(),
            column_indexes: self.column_indexes.as_ptr(),
            values: self.values.as_ptr()
        }
    }
}

pub enum Matrix<T> {
    Dense(Mutex<Arc<Array2<T>>>),
    Sparse(Mutex<Arc<CsrMatrix<T>>>)
}

pub enum LockedMatrix<T> {
    Dense(Arc<Array2<T>>),
    Sparse(Arc<CsrMatrix<T>>)
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
pub fn deterministic_weights(weights: &LockedMatrix<f32>) -> Matrix<isize> {
    let mut rng = thread_rng();
    match weights {
        LockedMatrix::Dense(m) => {
            let mut determ_weights = Array2::zeros((m.shape()[0], m.shape()[1]));
            for (idx, p) in m.indexed_iter() {
                determ_weights[idx] = event_prob_to_timespan(*p, &mut rng);
            }
            Matrix::Dense(Mutex::new(Arc::new(determ_weights)))
        },
        LockedMatrix::Sparse(sm) => {
            let mut determ_weights = CsrMatrix::new(sm.rows, sm.columns);
            let mut rng = thread_rng();
            for i in 0..sm.rows {
                determ_weights.cum_row_indexes.push(determ_weights.values.len());
                for j in sm.cum_row_indexes[i]..sm.cum_row_indexes[i+1] {
                    determ_weights.column_indexes.push(sm.column_indexes[j]);
                    determ_weights.values.push(event_prob_to_timespan(sm.values[j], &mut rng));
                }
            }
            determ_weights.cum_row_indexes.push(determ_weights.values.len());
            Matrix::Sparse(Mutex::new(Arc::new(determ_weights)))
        }
    }
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

    pub fn new_sparse_from_communities(communities: Vec<Vec<Node>>, intra_community_weight: f32, 
                                    inter_community_conn_prob: f32, inter_community_weight: f32) -> Graph {
        let total_nodes = communities.iter().fold(0, |n, c| n + c.len());
        let mut s_w = CsrMatrix::new(total_nodes, total_nodes);
        
        let mut cur_comm = 0;
        let mut comm_start = 0;
        for i in 0..total_nodes {
            s_w.cum_row_indexes.push(s_w.values.len());
            if i - comm_start >= communities[cur_comm].len() {
                comm_start = i;
                cur_comm += 1;
            }
            for j in 0..total_nodes {
                if j != i {
                    if j >= comm_start && j < comm_start + communities[cur_comm].len() {
                        s_w.values.push(intra_community_weight);
                        s_w.column_indexes.push(j)
                    } else if random::<f32>() < inter_community_conn_prob {
                        s_w.values.push(inter_community_weight);
                        s_w.column_indexes.push(j);
                    }
                }
            }
        }
        s_w.cum_row_indexes.push(s_w.values.len());

        Graph {
            nodes: communities.into_iter().flatten().collect(),
            weights: Matrix::Sparse(Mutex::new(Arc::new(s_w)))
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
