GPU Concurrent Agent-Level Contact Network Simulation

Basic simulations of infectious disease spread allowing for a variety of disease models.
The rate limiting step in the (stochastic) simulations is the calculation of the probabilities of new infections given a previous set of infectors and a contact graph.
We improve performance significantly by transforming the problem of calculating the probabilities of new infections into a generalized matrix vector multiplication, with the matrix given by the contact graph's adjacency matrix, the vector given by the current infectivities, and the generalized "multiplication" (1-(\*)(\*) instead of (*)(+).

We also include deterministic simulations, with "deterministic" referring to the fact that outcomes are fully determined by a transformed contact graph prior to calculation of simulation paths. 
The transformation of the (still stochastically generated) contact graph calculates the time to infection (d) for each edge i -> j. 
Such that, if i is infected on day 0, j will be infected on day d.
After the transformation, the disease path beginning with an initial infector can be calculated with the SSSP algorithm.

696