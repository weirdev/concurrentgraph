use ndarray::Array2;
use rand::prelude::*;

#[derive(Clone)]
#[derive(Copy)]
enum AgentStatus {
    Asymptomatic,
    //Symptomatic,
    //Quarintined(usize), //time left
    Dead
}

#[derive(Clone)]
#[derive(Copy)]
enum InfectionStatus {
    Infected(usize), // time left
    NotInfected(f32) // immunity
}

#[derive(Clone)]
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
    post_infection_immunity: f32
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

    fn simulate(&mut self, steps: usize, diseases: &[Disease]) {
        for _ in 0..steps {
            let mut new_nodes: Vec<Node> = Vec::new();
            // Transmissions
            for (nodenum, (node, adj_weights)) in self.nodes.iter().zip(self.weights.outer_iter()).enumerate() {
                let new_node = match node.status {
                    AgentStatus::Asymptomatic => { 
                        let mut died = false;
                        let mut new_infections: Vec<InfectionStatus> = Vec::new();
                        for (didx, dis) in diseases.iter().enumerate() {
                            let infection = match node.infections[didx] {
                                InfectionStatus::Infected(1) => {
                                    if random::<f32>() < dis.mortality_rate {
                                        died = true;
                                        InfectionStatus::Infected(0)
                                    } else {
                                        InfectionStatus::NotInfected(0.99)
                                    }
                                },
                                InfectionStatus::Infected(t) => InfectionStatus::Infected(t-1),
                                InfectionStatus::NotInfected(immunity) => {
                                    //println!("test1");
                                    let transmission = adj_weights.into_iter().enumerate()
                                        .filter(|(_, w)| **w != 0.0)
                                        //.inspect(|(i, _)| println!("i1 {}", *i))
                                        .filter(|(i, _)| match self.nodes[*i].infections[didx] {
                                            InfectionStatus::NotInfected(_) => false,
                                            InfectionStatus::Infected(_) => true
                                        })//.inspect(|(i, _)| println!("p1 {}", *i))
                                        .any(|(_, w)| random::<f32>() < (dis.transmission_rate * (1.0-immunity) * *w));
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
            println!("{} dead", self.dead_count());
            println!("{} infected", self.infected_count(0));
        }
    }
}

fn main() {
    let mut nodes: Vec<Node> = Vec::new();
    nodes.resize(100, Node {
        status: AgentStatus::Asymptomatic,
        infections: vec![InfectionStatus::NotInfected(0.1)]
    });
    let flu = Disease {
        name: "flu",
        transmission_rate: 0.3,
        mortality_rate: 0.1,
        infection_length: 10,
        post_infection_immunity: 0.99
    };
    nodes.push(Node {
        status: AgentStatus::Asymptomatic,
        infections: vec![InfectionStatus::Infected(flu.infection_length)]
    });
    let mut graph = Graph::new_uniform_from_nodes(nodes, 0.3);
    graph.simulate(200, &[flu]);
    println!("{} dead", graph.dead_count());
    println!("{} infected", graph.infected_count(0));
}
