# An NK Landscape Model

### Model Inspiration
This model is inspired by Lazer and Friedman in their 2007 paper [The Network Structure of Exploration and Exploitation](https://doi.org/10.2189/asqu.52.4.667).

This paper tests the impact that differing network structures have on the performance of a group in coming to an "optimal" solutions. The models have three different model types:
1. Linear 
2. Totally Connected
3. Random

Agents are randomly placed at a point in this communication network structure, and are also randomly placed at a solution point in the NK Landscape. The starting position in the NK Landscape is independent of the starting position in the communication network. There are `1000` randomly generated NK landscape spaces with the following parameters:
```python
N = 20
K = 5
population = 100
```

