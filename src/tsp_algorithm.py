"""
Abstract base class for TSP algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Optional


class TSPAlgorithm(ABC):
    """
    Abstract base class defining the interface for TSP solving algorithms.
    
    All TSP algorithm implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize the TSP algorithm with a distance matrix.
        
        Args:
            distance_matrix: Square matrix of distances between cities
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.best_tour: Optional[List[int]] = None
        self.best_distance: float = float('inf')
        self.history: Dict[str, List] = {
            'best_distances': [],
            'iteration_distances': [],
            'iterations': []
        }
    
    @abstractmethod
    def solve(self, **kwargs) -> Tuple[List[int], float]:
        """
        Solve the TSP instance.
        
        Args:
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Tuple of (best_tour, best_distance)
            - best_tour: List of city indices representing the optimal tour
            - best_distance: Total distance of the best tour found
        """
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize algorithm-specific data structures and parameters."""
        pass
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """
        Calculate the total distance of a given tour.
        
        Args:
            tour: List of city indices representing a tour
        
        Returns:
            Total distance of the tour
        """
        distance = 0.0
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            distance += self.distance_matrix[city_a, city_b]
        return distance
    
    def validate_tour(self, tour: List[int]) -> bool:
        """
        Validate that a tour visits all cities exactly once.
        
        Args:
            tour: List of city indices
        
        Returns:
            True if tour is valid, False otherwise
        """
        if len(tour) != self.num_cities:
            return False
        return len(set(tour)) == self.num_cities and all(0 <= city < self.num_cities for city in tour)
    
    def get_best_solution(self) -> Tuple[List[int], float]:
        """
        Get the best solution found so far.
        
        Returns:
            Tuple of (best_tour, best_distance)
        """
        if self.best_tour is None:
            raise ValueError("No solution found yet. Run solve() first.")
        return self.best_tour.copy(), self.best_distance
    
    def get_history(self) -> Dict[str, List]:
        """
        Get the optimization history.
        
        Returns:
            Dictionary containing:
            - 'best_distances': Best distance found at each iteration
            - 'iteration_distances': All distances evaluated per iteration
            - 'iterations': Iteration numbers
        """
        return self.history
    
    def reset(self) -> None:
        """Reset the algorithm state for a new run."""
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = {
            'best_distances': [],
            'iteration_distances': [],
            'iterations': []
        }
    
    def _update_best_solution(self, tour: List[int], distance: float) -> bool:
        """
        Update the best solution if a better one is found.
        
        Args:
            tour: Candidate tour
            distance: Distance of the candidate tour
        
        Returns:
            True if best solution was updated, False otherwise
        """
        if distance < self.best_distance:
            self.best_tour = tour.copy()
            self.best_distance = distance
            return True
        return False
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_cities={self.num_cities}, "
            f"best_distance={self.best_distance:.2f})"
        )
