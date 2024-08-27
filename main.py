from typing import List
import networkx as nx
from agent import Agent
from nklandscape import NKLandscape

# initialize parameters
num_simulations = 10
num_agents = 100
network_type = 'linear'

# initialize the NK Landscape and the initial agent solutions
N = 10
K = 2
nk_landscape = NKLandscape(N, K)
agents = [Agent(i, nk_landscape.generate_random_solution()) for i in range(num_agents)]

# initialize network based on the specified type
if network_type == 'linear':
    network = nx.path_graph(num_agents)
elif network_type == 'total':
    network = nx.complete_graph(num_agents)
elif network_type == 'random':
    p = 0.1  # Probability of edge creation
    network = nx.erdos_renyi_graph(num_agents, p)

# assign pre-generated solutions to each agent in the network
for i, agent in enumerate(agents):
    network.nodes[i]['agent'] = agent

# Main simulation loop
def simulate(network: nx.Graph) -> List[float]:
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

# Example: Access the solution string of a specific agent
example_agent = 0
print(f"Agent {example_agent} has solution: {network.nodes[example_agent]['solution']}")