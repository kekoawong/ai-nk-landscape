from typing import Dict
import numpy as np

class NKLandscape:
    """
    A class to represent an NK Landscape model.
    From Lazaar and Friedman: https://doi.org/10.2189/asqu.52.4.667
    
    Attributes:
        N (int): Number of components in the solution string.
        K (int): Number of dependencies for each component.
        landscape (np.ndarray): The fitness landscape as a numpy array.
        fitness_dict (Dict[str, float]): A dictionary mapping solution strings to their fitness values.
    """

    def __init__(self, N: int, K: int):
        """
        Initializes the NK landscape with the given N and K parameters.
        
        Args:
            N (int): The number of components in the solution string.
            K (int): The number of dependencies for each component.
        """
        self.N: int = N
        self.K: int = K
        self.landscape: np.ndarray = self._generate_landscape()
        self.fitness_dict: Dict[str, float] = {}

    def _generate_landscape(self) -> np.ndarray:
        """
        Generates the fitness landscape based on N and K randomly with values [0, 1).
        
        Returns:
            np.ndarray: A numpy array representing the fitness landscape. Dimensions are N * 2^(K+1)
        """
        return np.random.rand(self.N, 2 ** (self.K + 1))

    def get_fitness(self, solution: str) -> float:
        """
        Calculates the fitness of a given solution string.
        
        Args:
            solution (str): A binary string representing the solution vector. ex) "1010101010"
        
        Returns:
            float: The fitness value of the solution.
        """
        # for runtime efficiency, return solution if already calculated
        if solution in self.fitness_dict:
            return self.fitness_dict[solution]
        
        fitness: float = 0.0
        for i in range(self.N):
            index: int = self._get_index(i, solution)
            fitness += self.landscape[i, index]
        
        # fitness is the average of all the fitness indices
        fitness /= self.N
        self.fitness_dict[solution] = fitness
        return fitness

    def _get_index(self, i: int, solution: str) -> int:
        """
        Helper function to get the index in the landscape for a given component.
        
        Args:
            i (int): The index of the component in the solution string.
            solution (str): The binary solution string.

        Example:
            N = 5, K = 3, solution = "10101"
            indices will be 101 --> index will be 5 (since base 2 int)
        
        Returns:
            int: The index corresponding to the component and its dependencies.
        """
        indices = [solution[(i + j) % self.N] for j in range(self.K + 1)]
        return int(''.join(indices), 2)

# Example Usage
if __name__ == "__main__":
    N = 10  # Number of components
    K = 2   # Dependencies per component
    
    nk_landscape = NKLandscape(N, K)
    solution = "1010101010"
    fitness = nk_landscape.get_fitness(solution)
    print(f"Fitness of solution {solution}: {fitness}")
