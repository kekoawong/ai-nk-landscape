from modelpy_abm.main import AgentModel
import random

# Template initial data and timestep data functions


def generateInitialData():
    return {"x_location": random.randint(1, 50), "y_location": random.randint(1, 50)}


def generateTimestepData(model: AgentModel):
    graph = model.get_graph()
    for _node, node_data in graph.nodes(data=True):
        node_data["x_location"] += random.randint(-10, 10)

    model.set_graph(graph)


# Define AgentModel instance
model = AgentModel()

model["num_nodes"] = 7
model["default_bias"] = 0.15

# We can also define our parameters with this helper function
model.update_parameters({"num_nodes": 7, "default_bias": 0.15})


model.set_initial_data_function(generateInitialData)
model.set_timestep_function(generateTimestepData)

# Initialize the graph
model.initialize_graph()
# Run for loop for number of timesteps
timesteps = 100

for _ in range(timesteps):
    model.timestep()

# Print results
print(model.get_graph())
