use std::sync::{Arc, Mutex, MutexGuard};
use rand::prelude::*;
use std::time::SystemTime;

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
    enum LockedMatrixAndGpuAllocs<'a> {
        Dense((MutexGuard<'a, Arc<Array2<f32>>>, Option<NpmmvDenseGpuAllocations>)),
        Sparse((MutexGuard<'a, Arc<CsrMatrix<f32>>>, Option<NpmmvCsrGpuAllocations>))
    }

    let mat = match &mut graph.weights {
        Matrix::Dense(m) => {
            let lmat = m.lock().unwrap();
            let gpualloc = match mat_mul_func {
                MatMulFunction::GPU => {
                    let ga = npmmv_dense_gpu_allocate_safe(lmat.shape()[0], lmat.shape()[1]);
                    npmmv_gpu_set_dense_matrix_safe(&lmat, ga);
                    Some(ga)
                },
                _ => None
            };
            LockedMatrixAndGpuAllocs::Dense((lmat, gpualloc))
        },
        Matrix::Sparse(sm) => {
            let lmat = sm.lock().unwrap();
            let gpualloc = match mat_mul_func {
                MatMulFunction::GPU => {
                    let ga = npmmv_csr_gpu_allocate_safe(lmat.rows, lmat.columns, lmat.values.len());
                    npmmv_gpu_set_csr_matrix_safe(lmat.get_ptrs(), ga, lmat.rows, lmat.values.len());
                    Some(ga)
                },
                _ => None
            };
            LockedMatrixAndGpuAllocs::Sparse((lmat, gpualloc))
        }
    };

    let mut quarantined_nodes: Vec<Node> = Vec::new();

    for ts in 0..steps {
        let node_transmitivity: Vec<f32> = graph.nodes.iter().map(|n| match n.status {
            AgentStatus::Asymptomatic => match n.infections[0] {
                InfectionStatus::Infected(t) => 
                        (*diseases[0].shedding_fun)(t as isize) * diseases[0].transmission_rate,
                _ => 0.0,
            },
            AgentStatus::Dead => 0.0
        }).collect();

        let pr_no_infections = match mat_mul_func {
            MatMulFunction::SingleThreaded => match mat {
                LockedMatrixAndGpuAllocs::Dense(ref mga) => negative_prob_multiply_dense_matrix_vector_cpu_safe(1, mga.0.clone(), node_transmitivity).unwrap(),
                LockedMatrixAndGpuAllocs::Sparse(_) => panic!("Sparse matrices not implemented for CPU simulations")
            },
            MatMulFunction::MultiThreaded => match mat {
                LockedMatrixAndGpuAllocs::Dense(ref mga) => generic_mat_vec_mult_multi_thread(mga.0.clone(), node_transmitivity, Arc::new(|a, b| 1.0 - a*b), Arc::new(|a,b| a*b), 1.0).unwrap(),
                LockedMatrixAndGpuAllocs::Sparse(_) => panic!("Sparse matrices not implemented for CPU simulations")
            },
            MatMulFunction::GPU => {
                match mat {
                    LockedMatrixAndGpuAllocs::Dense(ref mga) => match mga.1 {
                        Some(ga) => {
                            npmmv_gpu_set_in_vector_safe(node_transmitivity, NpmmvAllocations::Dense(ga));
                            npmmv_dense_gpu_compute_safe(ga, mga.0.shape()[0], mga.0.shape()[1]);
                            npmmv_gpu_get_out_vector_safe(NpmmvAllocations::Dense(ga), mga.0.shape()[1])
                        },
                        None => panic!("GPU should be allocated at this point")
                    },
                    LockedMatrixAndGpuAllocs::Sparse(ref smga) => match smga.1 {
                        Some(ga) => {
                            npmmv_gpu_set_in_vector_safe(node_transmitivity, NpmmvAllocations::Sparse(ga));
                            npmmv_csr_gpu_compute_safe(ga, smga.0.rows, 1);
                            let v = npmmv_gpu_get_out_vector_safe(NpmmvAllocations::Sparse(ga), smga.0.rows);
                            v
                        },
                        None => panic!("GPU should be allocated at this point")
                    } 
                }
            }
        };

        quarantined_nodes.iter_mut().for_each(|n| match n.infections[0] {
            InfectionStatus::Infected(1) => {
                n.infections[0] = InfectionStatus::NotInfected(1.0);
                if random::<f32>() < diseases[0].mortality_rate {
                    println!("Quarantined patient died");
                }
            },
            InfectionStatus::Infected(t) => n.infections[0] = InfectionStatus::Infected(t-1),
            InfectionStatus::NotInfected(_) => ()
        });

        let mut dead = 0;
        let mut infected = 0;
        graph.nodes = graph.nodes.iter().zip(pr_no_infections).map(|(n, nipr)| match n.status {
            AgentStatus::Asymptomatic => {
                let mut died = false;
                let infstatus = match n.infections[0] {
                    InfectionStatus::Infected(1) => 
                        if random::<f32>() < diseases[0].mortality_rate {
                            died = true;
                            infected += 1;
                            InfectionStatus::Infected(0)
                        } else {
                            InfectionStatus::NotInfected(diseases[0].post_infection_immunity)
                        },
                    InfectionStatus::Infected(t) => {
                        if diseases[0].infection_length - 3 == t {
                            if random::<f32>() < 0.5 {
                                quarantined_nodes.push(n.clone());
                            }
                        }
                        infected += 1;
                        InfectionStatus::Infected(t-1)
                    },
                    InfectionStatus::NotInfected(immunity) => {
                        //println!("ilpr: {}, imm: {}", nipr, immunity);
                        // TODO: Modify weight matrix to reflect immunity
                        if random::<f32>() < (1.0 - nipr) * (1.0 - immunity) {
                            //println!("inf");
                            infected += 1;
                            InfectionStatus::Infected(diseases[0].infection_length)
                        } else {
                            //println!("no inf");
                            n.infections[0]
                        }
                    }
                };
                Node {
                    status: if died {
                                dead += 1;
                                AgentStatus::Dead
                            } else {
                                AgentStatus::Asymptomatic
                            },
                    infections: vec![infstatus]
                }
            },
            AgentStatus::Dead => n.clone()
        }).collect();
        //println!("T{}: {} dead, {} infected", ts, dead, infected);
    }

    match mat {
        LockedMatrixAndGpuAllocs::Dense(mga) => match mga.1 {
            Some(ga) => npmmv_dense_gpu_free_safe(ga),
            None => ()
        },
        LockedMatrixAndGpuAllocs::Sparse(smga) => match smga.1 {
            Some(ga) => npmmv_csr_gpu_free_safe(ga),
            None => ()
        }     
    }
}

pub fn simulate_basic_mat_bfs_cpu(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mut determ_weights = match graph.deterministic_infection_weights(diseases[0]) {
        Matrix::Dense(_) => panic!("not implemented"),
        Matrix::Sparse(m) => (**m.lock().unwrap()).clone()
    };

    let mut infections: Vec<usize> = graph.nodes.iter().map(|n| match n.infections[0] {
        InfectionStatus::Infected(_) => 1,
        InfectionStatus::NotInfected(_) => 0
    }).collect();
    let start_time = SystemTime::now();
    for ts in 0..steps {
        for i in 0..infections.len() {
            if infections[i] == 1 {
                for c in determ_weights.cum_row_indexes[i]..determ_weights.cum_row_indexes[i+1] {
                    let d = determ_weights.values[c];
                    if d != 0 {
                        if d == 1 {
                            infections[determ_weights.column_indexes[c]] = 1;
                        }
                        determ_weights.values[c] -= 1;
                    }
                }
            }
        }
    }
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("CPU Ran in {} secs", runtime.as_secs());
    
    let infection_count: usize = infections.iter().sum();
    println!("{} infections", infection_count);
}

pub fn simulate_basic_mat_bfs_gpu(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {

    let determ_weights = graph.deterministic_infection_weights(diseases[0]);

    let determ_weights = match &graph.weights {
        Matrix::Dense(_) => panic!("not implemented"),
        Matrix::Sparse(sm) => {
            let sm = sm.lock().unwrap().clone();
            let mat_ptrs = sm.get_ptrs();
            let mut shedding_curve: Vec<f32> = Vec::new();
            for i in 0..diseases[0].infection_length {
                shedding_curve.push((*diseases[0].shedding_fun)((diseases[0].infection_length-i) as isize));
            }
            let immunities: Vec<f32> = graph.nodes.iter().map(|n| match n.infections[0] {
                InfectionStatus::Infected(_) => 1.0,
                InfectionStatus::NotInfected(immun) => immun
            }).collect();
            
            let determ_w_vals = graph_deterministic_weights_gpu_safe(mat_ptrs, sm.rows, sm.values.len(), immunities, shedding_curve, diseases[0].infection_length, diseases[0].transmission_rate);
        
            let mut determ_weights_m: CsrMatrix<isize> = CsrMatrix::new(sm.rows, sm.columns);
            determ_weights_m.cum_row_indexes = sm.cum_row_indexes.clone();
            determ_weights_m.column_indexes = sm.column_indexes.clone();
            determ_weights_m.values = determ_w_vals;

            Matrix::Sparse(Mutex::new(Arc::new(determ_weights_m)))
        }
    };


    let infections: Vec<usize> = graph.nodes.iter().map(|n| match n.infections[0] {
        InfectionStatus::Infected(_) => 1,
        InfectionStatus::NotInfected(_) => 0
    }).collect();

    enum LockedMatrixAndGpuAllocs {
        Dense((Arc<Array2<isize>>, ())),
        Sparse((Arc<CsrMatrix<isize>>, BfsCsrGpuAllocations))
    }

    let mut mat_allocs = match determ_weights {
        Matrix::Dense(m) => LockedMatrixAndGpuAllocs::Dense((m.lock().unwrap().clone(), ())),
        Matrix::Sparse(sm) => {
            let lsm = sm.lock().unwrap().clone();
            let ga = bfs_csr_gpu_allocate_safe(lsm.rows, lsm.values.len());
            
            bfs_gpu_set_csr_matrix_safe(lsm.get_ptrs(), ga, lsm.rows, lsm.values.len());
            
            bfs_gpu_set_in_vector_safe(infections, BfsAllocations::Sparse(ga));

            LockedMatrixAndGpuAllocs::Sparse((lsm, ga))
        }
    };

    let start_time = SystemTime::now();
    for ts in 0..steps {
        mat_allocs = match mat_allocs {
            LockedMatrixAndGpuAllocs::Dense(_) => panic!("Dense operations not implemented yet"),
            LockedMatrixAndGpuAllocs::Sparse(mga) => {
                bfs_csr_gpu_compute_safe(mga.1, mga.0.rows);
                let new_ga = match bfs_gpu_swap_in_out_vector_refs(BfsAllocations::Sparse(mga.1)) {
                    BfsAllocations::Sparse(ga) => ga,
                    _ => panic!("Must be sparse")
                };
                LockedMatrixAndGpuAllocs::Sparse((mga.0, new_ga))
            }
        };
    }
    let runtime = SystemTime::now().duration_since(start_time)
        .expect("Time went backwards");
    println!("GPU Ran in {} secs", runtime.as_secs());

    let out_infections = match mat_allocs {
        LockedMatrixAndGpuAllocs::Dense(_) => panic!("Dense operations not implemented yet"),
        LockedMatrixAndGpuAllocs::Sparse(mga) => bfs_gpu_get_out_vector_safe(BfsAllocations::Sparse(mga.1), mga.0.rows)
    };
    
    let infection_count: usize = out_infections.iter().sum();
    println!("{} infections", infection_count);
}

pub fn simulate_basic_looped_deterministic(graph: &mut Graph, steps: usize, diseases: &[&Disease]) {
    let mat = match &graph.weights {
        Matrix::Dense(m) => m.lock().unwrap(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
    let mut determ_weights = match deterministic_contact_weights(&LockedMatrix::Dense(mat.clone())) {
        Matrix::Dense(m) => (*m.lock().unwrap().clone()).clone(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mut determ_weights = match deterministic_contact_weights(&LockedMatrix::Dense(mat.clone())) {
        Matrix::Dense(m) => (*m.lock().unwrap().clone()).clone(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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
    let mut determ_weights = match deterministic_contact_weights(&LockedMatrix::Dense(mat.clone())) {
        Matrix::Dense(m) => (*m.lock().unwrap().clone()).clone(),
        Matrix::Sparse(_) => panic!("Sparse matrices not implemented yet")
    };
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