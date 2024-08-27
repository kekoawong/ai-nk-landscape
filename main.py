from typing import List
import networkx as nx
import pandas as pd
from agent import Agent
from nklandscape import NKLandscape

# initialize parameters
num_simulations = 10
num_agents = 100
network_type = 'random'
p = 0.4  # Probability of edge creation, only for a random graph
velocity = 0.5
_error = 1 # UNIMPLEMENTED

# initialize the NK Landscape and the initial agent solutions
N = 10
K = 2

# Main simulation loop
def simulate() -> List[float]:
    nk_landscape = NKLandscape(N, K)
    agents = [Agent(i, nk_landscape.generate_random_solution(), velocity=velocity) for i in range(num_agents)]

    # initialize network based on the specified type
    if network_type == 'linear':
        network = nx.path_graph(num_agents)
    elif network_type == 'total':
        network = nx.complete_graph(num_agents)
    elif network_type == 'random':
        network = nx.erdos_renyi_graph(num_agents, p)

    # assign pre-generated solutions to each agent in the network
    for i, agent in enumerate(agents):
        network.nodes[i]['agent'] = agent

    # Store average fitness for each timestep
    avg_fitness_per_timestep = []

    # run until convergence
    converged = False
    while not converged:
        converged = True
        total_fitness = 0.0

        # update all the agents, set convergence bool
        for agent in agents:
            neighbors = [network.nodes[neighbor]['agent'] for neighbor in network.neighbors(agent.id)]
            current_solution: str = agent.solution
            # QUESTION: Mid-round updates may give an advantage to an agent seeing the updates of other agent to make their own decisions?
            agent.update_solution(neighbors, nk_landscape)
            if agent.solution != current_solution:
                converged = False
            
            # add agent fitness to collect metrics
            total_fitness += nk_landscape.get_fitness(agent.solution)

        # append average fitness
        avg_fitness_per_timestep.append(total_fitness / num_agents)
    
    return avg_fitness_per_timestep

# Run simulations and store results
results = []
for simulation_num in range(num_simulations):
    avg_fitness_per_timestep = simulate()
    print(f'Simulation number ${simulation_num} and values ${avg_fitness_per_timestep}')
    for timestep, avg_fitness in enumerate(avg_fitness_per_timestep):
        results.append({'Simulation': simulation_num, 'Timestep': timestep, 'Avg_Fitness': avg_fitness})

# Convert results to DataFrame
df = pd.DataFrame(results)
print(df)

# Save DataFrame to CSV
df.to_csv('average_fitness_per_timestep.csv', index=False)