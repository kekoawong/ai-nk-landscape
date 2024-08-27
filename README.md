# AI's impact on epistemic landscapes
This code implements an NK Landscape model that illustrates AI's impact on a population's performance in finding an ideal solution.

### Model Inspiration
This model is inspired by Lazer and Friedman in their 2007 paper [The Network Structure of Exploration and Exploitation](https://doi.org/10.2189/asqu.52.4.667).

This paper tests the impact that differing network structures have on the performance of a group in coming to an "optimal" solutions. The models have three different network types:
1. Linear 
2. Totally Connected
3. Random (p=0.4 for now, which describes the probability that an edge will be formed between two nodes)

Agents are randomly placed at a point in this communication network structure, and are also randomly placed at a solution point in the NK Landscape. The starting position in the NK Landscape is independent of the starting position in the communication network. 

There are `1000` randomly generated NK landscape spaces (for each simulation) with the following parameters:
```python
N = 20
K = 5
population = 100
```

Lazer and Friedman also introduce the idea of **velocity** (which refers to the probability "that an agent will look at his or her network each round") and **error** (which refers to how accurrately an agent will copy their neighbors solution if it is better than their own). For now, we have initialized these parameters to the following:
```python
velocity = 0.5
error = 1
```

