from typing import Dict
import numpy as np

class NKLandscape:
    """
    A class to represent an NK Landscape model.
    From Lazaar and Friedman: https://doi.org/10.2189/asqu.52.4.667
    
    Attributes:
        N (int): Number of components in the solution string.
        K (int): Number of dependencies for each component.
        landscape (np.ndarray): The fitness landscape as a 2D numpy array.
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
        Generates the fitness landscape based on N and K.
        
        Returns:
            np.ndarray: A 2D numpy array representing the fitness landscape.
        """
        return np.random.rand(self.N, 2 ** (self.K + 1))

    def _calculate_fitness(self, solution: str) -> float:
        """
        Calculates the fitness of a given solution string.
        
        Args:
            solution (str): A binary string representing the solution. ex) "1010101010"
        
        Returns:
            float: The fitness value of the solution.
        """
        if solution in self.fitness_dict:
            return self.fitness_dict[solution]
        
        fitness: float = 0.0
        for i in range(self.N):
            index: int = self._get_index(i, solution)
            fitness += self.landscape[i, index]
        
        fitness /= self.N
        self.fitness_dict[solution] = fitness
        return fitness

    def _get_index(self, i: int, solution: str) -> int:
        """
        Helper function to get the index in the landscape for a given component.
        
        Args:
            i (int): The index of the component in the solution string.
            solution (str): The binary solution string.
        
        Returns:
            int: The index corresponding to the component and its dependencies.
        """
        indices = [solution[(i + j) % self.N] for j in range(self.K + 1)]
        return int(''.join(indices), 2)

    def get_fitness(self, solution: str) -> float:
        """
        Public method to get the fitness of a given solution.
        
        Args:
            solution (str): A binary string representing the solution.
        
        Returns:
            float: The fitness value of the solution.
        """
        return self._calculate_fitness(solution)

# Example Usage
if __name__ == "__main__":
    N = 10  # Number of components
    K = 2   # Dependencies per component
    
    nk_landscape = NKLandscape(N, K)
    solution = "1010101010"
    fitness = nk_landscape.get_fitness(solution)
    print(f"Fitness of solution {solution}: {fitness}")
