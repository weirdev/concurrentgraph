use std::sync::Arc;
use rand::prelude::*;

use ndarray::{Array, Array2, Axis};

use concurrentgraph_cuda_sys::*;

mod graph;
pub use graph::*;

pub fn simulate_basic_looped_stochastic(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };

    let mut gpu_allocations: Option<NpmmvDenseGpuAllocations> = None;
    match mat_mul_func {
        MatMulFunction::GPU => {
            let ga = npmmv_dense_gpu_allocate_safe(mat.shape()[0], mat.shape()[1]);
            npmmv_gpu_set_dense_matrix_safe(&mat, ga);
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

        let nodetrans_copy = node_transmitivity.clone();
        let pr_no_infections = match mat_mul_func {
            MatMulFunction::SingleThreaded => negative_prob_multiply_dense_matrix_vector_cpu_safe(1, mat.clone(), nodetrans_copy).unwrap(),
            MatMulFunction::MultiThreaded => generic_mat_vec_mult_multi_thread(mat.clone(), nodetrans_copy, Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).unwrap(),
            MatMulFunction::GPU => {
                match gpu_allocations {
                    Some(ga) => {
                        npmmv_gpu_set_in_vector_safe(nodetrans_copy, ga);
                        npmmv_dense_gpu_compute_safe(ga, mat.shape()[0], mat.shape()[1]);
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
        Some(ga) => npmmv_dense_gpu_free_safe(ga),
        None => ()
    }
}

pub fn simulate_looped_bfs(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mut mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mut mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap().clone(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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