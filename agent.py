from typing import List
import numpy as np
from nklandscape import NKLandscape

class Agent:
    def __init__(self, id: int, initial_solution: str) -> None:
        self.id: int = id
        self.solution: str = initial_solution
    
    def mutate_solution(self) -> str:
        """Randomly flips a bit in the solution string."""
        solution_list: List[str] = list(self.solution)
        idx: int = np.random.randint(len(solution_list))
        solution_list[idx] = '1' if solution_list[idx] == '0' else '0'
        self.solution = ''.join(solution_list)
        return self.solution
    
    def update_solution(self, neighbors: List['Agent'], landscape: NKLandscape) -> str:
        """Updates the agent's solution based on the best neighbor or mutation."""
        current_fitness: float = landscape.get_fitness(self.solution)
        
        # Check neighbors' solutions
        best_neighbor_solution: str = self.solution
        for neighbor in neighbors:
            neighbor_fitness: float = landscape.get_fitness(neighbor.solution)
            if neighbor_fitness > landscape.get_fitness(best_neighbor_solution):
                best_neighbor_solution = neighbor.solution

        if landscape.get_fitness(best_neighbor_solution) > current_fitness:
            self.solution = best_neighbor_solution
        else:
            # Mutate and compare
            original_solution: str = self.solution
            self.mutate_solution()
            if landscape.get_fitness(self.solution) <= current_fitness:
                self.solution = original_solution
        return self.solution