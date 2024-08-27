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
    p = 0.5  # Probability of edge creation
    network = nx.erdos_renyi_graph(num_agents, p)

# assign pre-generated solutions to each agent in the network
for i, agent in enumerate(agents):
    network.nodes[i]['agent'] = agent

# loop through each timestep
# for each timestep, each agent will look at the value of their neighbor's solutions
# if so, they change their solution completely copies the solution of that neighbor
# if not, they will randomly change a digit in their current solution, compare solutions and pick the larger one

# keep looping through until all of the population has converged to a single solution

# Example: Access the solution string of a specific agent
example_agent = 0
print(f"Agent {example_agent} has solution: {network.nodes[example_agent]['solution']}")