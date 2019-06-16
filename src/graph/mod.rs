use std::io;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array, Array2, Axis};
use rand::prelude::*;
use rand::distributions::{Poisson, Gamma};

pub mod external_adaptor;

use external_adaptor::*;

#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
enum AgentStatus {
    Asymptomatic,
    //Symptomatic,
    //Quarintined(usize), //time left
    Dead // TODO: This can be replaced with an Asymptomatic node that is 100% immune
}

#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
enum InfectionStatus {
    Infected(usize), // time left
    NotInfected(f32) // immunity
}

#[derive(Clone)]
#[derive(Debug)]
pub struct Node {
    status: AgentStatus,
    infections: Vec<InfectionStatus>
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub weights: Mutex<Arc<Array2<f32>>>
}

pub enum MatMulFunction {
    SingleThreaded,
    MultiThreaded,
    GPU
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

pub fn generic_mat_vec_mult_multi_thread<'a, D>(mat_lock: &Mutex<Arc<Array2<D>>>, vector: Vec<D>, 
        op1: Arc<(Fn(&D, &D) -> D) + Sync + Send>, op2: Arc<(Fn(&D, &D) -> D) + Sync + Send>, initial_val: D) 
            -> Result<Vec<D>, &'a str> 
        where D: Copy + Sync + Send + 'static {
    let mat = mat_lock.lock().unwrap();
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

fn event_prob_to_timespan(p: f32, rng: &mut ThreadRng) -> isize {
    if p > 0.0 {
        let y = rng.sample(Gamma::new(1.0, ((1.0 - p) / p) as f64));
        rng.sample(Poisson::new(y)) as isize}
    else {-1}
}

// weights elements = probability of meeting at a given timestep
// output elements = number of timesteps until row meets column
// Sampling negative binomial distribution in same manner as random_negative_binomial() in
// https://github.com/numpy/numpy/blob/master/numpy/random/src/distributions/distributions.c
fn deterministic_weights(weights: &Array2<f32>) -> Array2<isize> {
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
            weights: Mutex::new(Arc::new(Array2::from_elem((n, n), 1.0)))
        }
    }

    fn new_uniform_from_nodes(nodes: Vec<Node>, connection_weights: f32) -> Graph {
        let mut w = Array2::from_elem((nodes.len(), nodes.len()), connection_weights);
        remove_weight_mat_self_refs(&mut w);
        Graph {
            weights: Mutex::new(Arc::new(w)),
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

pub fn simulate_basic_looped_stochastic(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mut mat = graph.weights.lock().unwrap();
    for ts in 0..steps {
        let mut new_nodes: Vec<Node> = Vec::new();
        // For each node and its outgoing edges
        for (nodenum, (node, adj_weights)) in graph.nodes.iter().zip(mat.outer_iter()).enumerate() {
            let new_node = match node.status {
                // Node is asymptomatic (just means alive for basic simulation),
                // check if it gets infected by a disease, or dies from a disease
                AgentStatus::Asymptomatic => { 
                    let mut died = false;
                    let mut new_infections: Vec<InfectionStatus> = Vec::new();
                    // For each disease, check if the disease is passed to node,
                    // or if node dies from the disease
                    for (didx, dis) in diseases.iter().enumerate() {
                        let infection = match node.infections[didx] {
                            // Infected, decrement counter, if would go to zero, check for death
                            InfectionStatus::Infected(1) => {
                                if random::<f32>() < dis.mortality_rate {
                                    died = true;
                                    InfectionStatus::Infected(0)
                                } else {
                                    InfectionStatus::NotInfected(dis.post_infection_immunity)
                                }
                            },
                            InfectionStatus::Infected(t) => InfectionStatus::Infected(t-1),
                            // Not infected, check for transmission
                            InfectionStatus::NotInfected(immunity) => {
                                let transmission = adj_weights.into_iter().enumerate()
                                    .filter(|(_, w)| **w != 0.0)
                                    .filter_map(|(i, w)| match graph.nodes[i].infections[didx] {
                                        InfectionStatus::NotInfected(_) => None,
                                        InfectionStatus::Infected(t) => Some((i, w, t))
                                    })
                                    .any(|(_, w, t)| random::<f32>() < (dis.transmission_rate * (*(*dis).shedding_fun)(t as isize) * (1.0-immunity) * *w));
                                if transmission {
                                    InfectionStatus::Infected(dis.infection_length)
                                } else {
                                    node.infections[didx]
                                }
                            }
                        };
                        new_infections.push(infection);
                    }
                    Node {
                        status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                        infections: new_infections
                    }
                },
                AgentStatus::Dead => node.clone()
            };
            new_nodes.push(new_node);
        }
        graph.nodes = new_nodes;
        //println!("T{}: {} dead, {} infected", t, self.dead_count(), self.infected_count(0));
    }
}

pub fn simulate_basic_mat_stochastic(graph: &mut Graph, steps: usize, diseases: &[&Disease], mat_mul_func: MatMulFunction) {
    //let mut node_transmitivity = Array2::<f32>::zeros((self.nodes.len(), self.nodes.len()));
    let mut gpu_allocations: Option<NpmmvGpuAllocations> = None;
    match mat_mul_func {
        MatMulFunction::GPU => {
            let mat = graph.weights.lock().unwrap();
            let ga = npmmv_gpu_allocate_safe(mat.shape()[0], mat.shape()[1]);
            npmmv_gpu_set_matrix_safe(&mat, ga);
            gpu_allocations = Some(ga);
        },
        _ => ()
    }

    for ts in 0..steps {
        let node_transmitivity: Vec<f32> = graph.nodes.iter().map(|n| match n.status {
            AgentStatus::Asymptomatic => match n.infections[0] {
                InfectionStatus::Infected(t) => 
                        (*diseases[0].shedding_fun)(t as isize) * diseases[0].transmission_rate,
                _ => 0.0,
            },
            AgentStatus::Dead => 0.0
        }).collect();
        
        //hello_safe();

        let nodetrans_copy = node_transmitivity.clone();
        let pr_no_infections = match mat_mul_func {
            MatMulFunction::SingleThreaded => negative_prob_multiply_matrix_vector_cpu_safe(1, &graph.weights, nodetrans_copy).unwrap(),
            MatMulFunction::MultiThreaded => generic_mat_vec_mult_multi_thread(&graph.weights, nodetrans_copy, Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).unwrap(),
            MatMulFunction::GPU => {
                match gpu_allocations {
                    Some(ga) => {
                        npmmv_gpu_set_in_vector_safe(nodetrans_copy, ga);
                        let mat = graph.weights.lock().unwrap();
                        npmmv_gpu_compute_safe(ga, mat.shape()[0], mat.shape()[1]);
                        npmmv_gpu_get_out_vector_safe(ga, mat.shape()[1])
                    },
                    None => panic!("Should never reach here")
                }
            }
        };

        //println!("t: {}, n_infection_prs: {:?}", t, log_pr_infections.iter().map(|l| (2.0 as f32).powf(*l)).collect::<Vec<f32>>());

        graph.nodes = graph.nodes.iter().zip(pr_no_infections).map(|(n, nipr)| match n.status {
            AgentStatus::Asymptomatic => {
                let mut died = false;
                let infstatus = match n.infections[0] {
                    InfectionStatus::Infected(1) => 
                        if random::<f32>() < diseases[0].mortality_rate {
                            died = true;
                            InfectionStatus::Infected(0)
                        } else {
                            InfectionStatus::NotInfected(diseases[0].post_infection_immunity)
                        },
                    InfectionStatus::Infected(t) => InfectionStatus::Infected(t-1),
                    InfectionStatus::NotInfected(immunity) => {
                        //println!("ilpr: {}, imm: {}", nipr, immunity);
                        // TODO: Modify weight matrix to reflect immunity
                        if random::<f32>() < (1.0 - nipr) * (1.0 - immunity) {
                            //println!("inf");
                            InfectionStatus::Infected(diseases[0].infection_length)
                        } else {
                            //println!("no inf");
                            n.infections[0]
                        }
                    }
                };
                Node {
                    status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                    infections: vec![infstatus]
                }
            },
            AgentStatus::Dead => n.clone()
        }).collect();
        //println!("T{}: {} dead, {} infected", ts, self.dead_count(), self.infected_count(0));
    }

    match gpu_allocations {
        Some(ga) => npmmv_gpu_free_safe(ga),
        None => ()
    }
}

pub fn simulate_looped_bfs(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = graph.weights.lock().unwrap();
    for ts in 1..steps+1 {
        let mut new_nodes: Vec<Node> = Vec::new();
        let nodes = &mut graph.nodes;
        for (nodenum, (node, ref mut adj_weights)) in nodes.iter().zip(mat.outer_iter()).enumerate() {
        let new_node = match node.status {
                // Node is asymptomatic (just means alive for basic simulation),
                // check if it gets infected by a disease, or dies from a disease
                AgentStatus::Asymptomatic => { 
                    let mut died = false;
                    let mut new_infections: Vec<InfectionStatus> = Vec::new();
                    // For each disease, check if the disease is passed to node,
                    // or if node dies from the disease
                    let infection = match node.infections[0] {
                        // Infected, decrement counter, if would go to zero, check for death
                        InfectionStatus::Infected(t) => {
                            if ts - t == diseases[0].infection_length {
                                died = true;
                            }
                            InfectionStatus::Infected(t)
                        },
                        // Not infected, check for transmission
                        InfectionStatus::NotInfected(immunity) => {
                            let incoming_infectors = adj_weights.into_iter().enumerate()
                                .filter(|(_, w)| **w != 0.0)
                                .filter_map(|(i, w)| match nodes[i].infections[0] {
                                    InfectionStatus::NotInfected(_) => None,
                                    InfectionStatus::Infected(t) => Some((i, w, t))
                                });
                            
                            let transmission = incoming_infectors.fold(false, |infection, (_, w, t)| 
                                infection || random::<f32>() < (diseases[0].transmission_rate * (*(*diseases[0]).shedding_fun)((ts - t) as isize) * (1.0-immunity) * *w)
                            );
                            if transmission {
                                InfectionStatus::Infected(ts)
                            } else {
                                node.infections[0]
                            }
                        }
                    };
                    new_infections.push(infection);

                    Node {
                        status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                        infections: new_infections
                    }
                },
                AgentStatus::Dead => node.clone()
            };
            new_nodes.push(new_node);
        }
        graph.nodes = new_nodes;
        println!("T{}: {} dead, {} infected", ts, graph.dead_count(), graph.infected_count(0));
    }
}

pub fn simulate_basic_looped_deterministic(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = graph.weights.lock().unwrap();
    let mut determ_weights = deterministic_weights(&mat);
    for t in 0..steps {
        let mut new_nodes: Vec<Node> = Vec::new();
        // For each node and its outgoing edges
        let nodes = &mut graph.nodes;
        for (nodenum, (node, ref mut adj_weights)) in nodes.iter().zip(determ_weights.outer_iter_mut()).enumerate() {
            let new_node = match node.status {
                // Node is asymptomatic (just means alive for basic simulation),
                // check if it gets infected by a disease, or dies from a disease
                AgentStatus::Asymptomatic => { 
                    let mut died = false;
                    let mut new_infections: Vec<InfectionStatus> = Vec::new();
                    // For each disease, check if the disease is passed to node,
                    // or if node dies from the disease
                    for (didx, dis) in diseases.iter().enumerate() {
                        let infection = match node.infections[didx] {
                            // Infected, decrement counter, if would go to zero, check for death
                            InfectionStatus::Infected(1) => {
                                if random::<f32>() < dis.mortality_rate {
                                    died = true;
                                    InfectionStatus::Infected(0)
                                } else {
                                    InfectionStatus::NotInfected(dis.post_infection_immunity)
                                }
                            },
                            InfectionStatus::Infected(t) => InfectionStatus::Infected(t-1),
                            // Not infected, check for transmission
                            InfectionStatus::NotInfected(immunity) => {
                                let incoming_infectors = adj_weights.into_iter().enumerate()
                                    .filter(|(_, w)| **w != -1)
                                    .filter_map(|(i, w)| match nodes[i].infections[didx] {
                                        InfectionStatus::NotInfected(_) => None,
                                        InfectionStatus::Infected(t) => Some((i, w, t))
                                    });
                                
                                let transmission = incoming_infectors.fold(false, |infection, (_, w, t)| {
                                        let inf: bool;
                                        if *w == 0 {
                                            inf = infection || random::<f32>() < (dis.transmission_rate * (*(*dis).shedding_fun)(t as isize) * (1.0-immunity))
                                        } else {
                                            inf = infection;
                                        }
                                        *w -= 1;
                                        inf
                                    });
                                if transmission {
                                    InfectionStatus::Infected(dis.infection_length)
                                } else {
                                    node.infections[didx]
                                }
                            }
                        };
                        new_infections.push(infection);
                    }
                    Node {
                        status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                        infections: new_infections
                    }
                },
                AgentStatus::Dead => node.clone()
            };
            new_nodes.push(new_node);
        }
        graph.nodes = new_nodes;
        println!("T{}: {} dead, {} infected", t, graph.dead_count(), graph.infected_count(0));
    }
}

pub fn simulate_simplistic_mat_deterministic(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = graph.weights.lock().unwrap();
    let mut determ_weights = deterministic_weights(&mat);
    for t in 0..steps {
        //println!("t: {}, n_infection_prs: {:?}", t, log_pr_infections.iter().map(|l| (2.0 as f32).powf(*l)).collect::<Vec<f32>>());
        graph.nodes = graph.nodes.iter().enumerate().map(|(i, n)| match n.status {
            AgentStatus::Asymptomatic => {
                let mut died = false;
                let infstatus = match n.infections[0] {
                    InfectionStatus::Infected(0) => InfectionStatus::Infected(0),
                    InfectionStatus::Infected(t) => {
                        if t == 1 {
                            if random::<f32>() < diseases[0].mortality_rate {
                                died = true;
                            }
                        }
                        for r in 0..determ_weights.shape()[0] {
                            determ_weights[(r, i)] -= 1;
                        }
                        InfectionStatus::Infected(t-1)
                    },
                    InfectionStatus::NotInfected(_) => {
                        let mut inf = n.infections[0];
                        for c in 0..determ_weights.shape()[1] {
                            if determ_weights[(i, c)] == 0 {
                                inf = InfectionStatus::Infected(diseases[0].infection_length);
                            }
                        }
                        inf
                    }
                };
                Node {
                    status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                    infections: vec![infstatus]
                }
            },
            AgentStatus::Dead => n.clone()
        }).collect();
        println!("T{}: {} dead, {} infected", t, graph.dead_count(), graph.infected_count(0));
    }
}

pub fn simulate_basic_looped_deterministic_shedding_incorrect(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = graph.weights.lock().unwrap();
    let mut determ_weights = deterministic_weights(&mat);
    for t in 0..steps {
        let mut new_nodes: Vec<Node> = Vec::new();
        // For each node and its outgoing edges
        let mut rng = thread_rng();
        let mut shedding_time: Vec<isize> = graph.nodes.iter().map(|n| match n.status {
            AgentStatus::Asymptomatic => match n.infections[0] {
                InfectionStatus::Infected(t) => event_prob_to_timespan((diseases[0].shedding_fun)(t as isize), &mut rng),
                InfectionStatus::NotInfected(_) => -1
            },
            AgentStatus::Dead => -1
        }).collect();
        let nodes = &mut graph.nodes;
        for (nodenum, (node, ref mut adj_weights)) in nodes.iter().zip(determ_weights.outer_iter_mut()).enumerate() {
            let new_node = match node.status {
                // Node is asymptomatic (just means alive for basic simulation),
                // check if it gets infected by a disease, or dies from a disease
                AgentStatus::Asymptomatic => { 
                    let mut died = false;
                    let mut new_infections: Vec<InfectionStatus> = Vec::new();
                    // For each disease, check if the disease is passed to node,
                    // or if node dies from the disease
                    for (didx, dis) in diseases.iter().enumerate() {
                        let infection = match node.infections[didx] {
                            // Infected, decrement counter, if would go to zero, check for death
                            InfectionStatus::Infected(1) => {
                                if random::<f32>() < dis.mortality_rate {
                                    died = true;
                                    InfectionStatus::Infected(0)
                                } else {
                                    InfectionStatus::NotInfected(dis.post_infection_immunity)
                                }
                            },
                            InfectionStatus::Infected(t) => InfectionStatus::Infected(t-1),
                            // Not infected, check for transmission
                            InfectionStatus::NotInfected(immunity) => {
                                let incoming_infectors = adj_weights.into_iter().enumerate()
                                    .filter(|(_, w)| **w != -1)
                                    .filter_map(|(i, w)| match nodes[i].infections[didx] {
                                        InfectionStatus::NotInfected(_) => None,
                                        InfectionStatus::Infected(t) => Some((i, w, t))
                                    });
                                
                                let transmission = incoming_infectors.fold(false, |infection, (_, w, t)| {
                                        let mut inf = false;
                                        if *w == 0 {
                                            if shedding_time[nodenum] == 0 {
                                                inf = true;
                                            }
                                            shedding_time[nodenum] -= 1;
                                        } else {
                                            inf = infection;
                                        }
                                        *w -= 1;
                                        inf
                                    });
                                if transmission {
                                    InfectionStatus::Infected(dis.infection_length)
                                } else {
                                    node.infections[didx]
                                }
                            }
                        };
                        new_infections.push(infection);
                    }
                    Node {
                        status: if died {
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                        infections: new_infections
                    }
                },
                AgentStatus::Dead => node.clone()
            };
            new_nodes.push(new_node);
        }
        graph.nodes = new_nodes;
        println!("T{}: {} dead, {} infected", t, graph.dead_count(), graph.infected_count(0));
    }
}