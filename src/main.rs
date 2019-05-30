use std::io;

#[macro_use(array)]
extern crate ndarray;

use ndarray::{Array, Array2, Axis};
use rand::prelude::*;
use rand::distributions::{Poisson, Gamma};

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
struct Node {
    status: AgentStatus,
    infections: Vec<InfectionStatus>
}

struct Graph {
    nodes: Vec<Node>,
    weights: Array2<f32>
}

fn remove_weight_mat_self_refs(weight_mat: &mut Array2<f32>) {
    for e in weight_mat.diag_mut() {
        *e = 0.0;
    }
}

struct Disease<'a> {
    name: &'a str,
    transmission_rate: f32,
    mortality_rate: f32,
    infection_length: usize,
    post_infection_immunity: f32,
    shedding_fun: Box<Fn(isize) -> f32>
}

impl Graph {
    fn new_connected(n: usize) -> Graph {
        Graph {
            //nodes: (0..n).map(|_| Node::new()).collect(),
            nodes: vec![Node {
                status: AgentStatus::Asymptomatic,
                infections: Vec::new()
            }; n],
            weights: Array2::from_elem((n, n), 1.0)
        }
    }

    fn new_uniform_from_nodes(nodes: Vec<Node>, connection_weights: f32) -> Graph {
        let mut w = Array2::from_elem((nodes.len(), nodes.len()), connection_weights);
        remove_weight_mat_self_refs(&mut w);
        Graph {
            weights: w,
            nodes: nodes
        }
    }

    fn dead_count(&self) -> usize {
        self.nodes.iter()
            .filter(|n| match n.status {
                AgentStatus::Dead => true,
                _ => false
            }).count()
    }

    fn infected_count(&self, disease: usize) -> usize {
        self.nodes.iter()
            .filter(|n| match n.infections[disease] {
                InfectionStatus::Infected(_) => true,
                _ => false
            }).count()
    }

    // weights elements = probability of meeting at a given timestep
    // output elements = number of timesteps until row meets column
    // Sampling negative binomial distribution in same manner as random_negative_binomial() in
    // https://github.com/numpy/numpy/blob/master/numpy/random/src/distributions/distributions.c
    fn deterministic_weights(&self, total_timesteps: usize) -> Array2<isize> {
        let mut rng = thread_rng();
        let mut determ_weights = Array2::zeros((self.weights.shape()[0], 2));
        for (idx, p) in self.weights.indexed_iter() {
            determ_weights[idx] = if *p > 0.0 {
                    let y = rng.sample(Gamma::new(1.0, ((1.0 - *p) / *p) as f64));
                    rng.sample(Poisson::new(y)) as isize}
                else {-1}
        }
        determ_weights
    }

    fn simulate_basic_looped_stochastic(&mut self, steps: usize, diseases: &[&Disease]) {
        for t in 0..steps {
            let mut new_nodes: Vec<Node> = Vec::new();
            // For each node and its outgoing edges
            for (nodenum, (node, adj_weights)) in self.nodes.iter().zip(self.weights.outer_iter()).enumerate() {
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
                                        .filter_map(|(i, w)| match self.nodes[i].infections[didx] {
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
            self.nodes = new_nodes;
            println!("T{}: {} dead, {} infected", t, self.dead_count(), self.infected_count(0));
        }
    }

    fn simulate_basic_mat_stochastic(&mut self, steps: usize, diseases: &[&Disease]) {
        for t in 0..steps {
            // TODO: matrix*diag can be more efficient (and no alloc of second matrix)
            let node_transmitivity_iter = self.nodes.iter().map(|n| match n.infections[0] {
                InfectionStatus::Infected(t) => (*diseases[0].shedding_fun)(t as isize) * diseases[0].transmission_rate,
                _ => 0.0
            });
            let mut node_transmitivity = Array2::<f32>::zeros((self.nodes.len(), self.nodes.len()));
            for (i, t) in node_transmitivity_iter.enumerate() {
                node_transmitivity[(i, i)] = t;
            }
            let mut pr_transmissions = self.weights.dot(&node_transmitivity);
            pr_transmissions.mapv_inplace(|p| (1.0 - p).log2());
            let log_pr_infections = pr_transmissions.sum_axis(Axis(1)).to_vec();

            //println!("t: {}, n_infection_prs: {:?}", t, log_pr_infections.iter().map(|l| (2.0 as f32).powf(*l)).collect::<Vec<f32>>());

            self.nodes = self.nodes.iter().zip(log_pr_infections).map(|(n, ilpr)| match n.status {
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
                            //println!("ilpr: {}, imm: {}", ilpr, immunity);
                            // TODO: Modify weight matrix to reflect immunity
                            if random::<f32>() < (1.0 - (2.0 as f32).powf(ilpr)) * (1.0 - immunity) {
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
            println!("T{}: {} dead, {} infected", t, self.dead_count(), self.infected_count(0));
        }
    }

    fn simulate_basic_looped_deterministic(&mut self, steps: usize, diseases: &[&Disease]) {
        for t in 0..steps {
            let mut new_nodes: Vec<Node> = Vec::new();
            // For each node and its outgoing edges
            for (nodenum, (node, adj_weights)) in self.nodes.iter().zip(self.weights.outer_iter()).enumerate() {
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
                                        .filter_map(|(i, w)| match self.nodes[i].infections[didx] {
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
            self.nodes = new_nodes;
            println!("T{}: {} dead, {} infected", t, self.dead_count(), self.infected_count(0));
        }
    }
}

fn new_sim_graph(n: usize, infection: &Disease)  -> Graph {
    let mut nodes: Vec<Node> = Vec::new();
    nodes.resize(n-1, Node {
        status: AgentStatus::Asymptomatic,
        infections: vec![InfectionStatus::NotInfected(0.1)]
    });
    
    nodes.push(Node {
        status: AgentStatus::Asymptomatic,
        infections: vec![InfectionStatus::Infected(infection.infection_length)]
    });
    Graph::new_uniform_from_nodes(nodes, 0.3)
}

fn main() -> io::Result<()> {
    let a = array![[1,2,3],[4,5,6]];
    println!("{:?}", a.sum_axis(Axis(1)).to_vec());

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let flu = Disease {
        name: "flu",
        transmission_rate: 0.3,
        mortality_rate: 0.1,
        infection_length: 10,
        post_infection_immunity: 0.99,
        shedding_fun: Box::new(|d| if d > 0 {1.0 / d as f32} else {0.0})
    };

    let mut graph = new_sim_graph(100, &flu);
    
    graph.simulate_basic_looped_stochastic(200, &[&flu]);
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let mut graph = new_sim_graph(100, &flu);
    graph.simulate_basic_mat_stochastic(200, &[&flu]);
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
    Ok(())
}
