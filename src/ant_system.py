"""
Ant System (AS) - The standard Ant Colony Optimization implementation.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.tsp_algorithm import TSPAlgorithm


class AntSystem(TSPAlgorithm):
    """
    Ant System (AS) algorithm for solving the Traveling Salesman Problem.
    
    This is the original ACO algorithm proposed by Dorigo et al. (1996).
    Ants construct solutions probabilistically based on pheromone trails
    and heuristic information (visibility).
    """
    
    def __init__(
        self,
        distance_matrix: np.ndarray,
        num_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 100.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the Ant System algorithm.
        
        Args:
            distance_matrix: Square matrix of distances between cities
            num_ants: Number of ants in the colony
            alpha: Pheromone importance factor (α)
            beta: Heuristic information importance factor (β)
            evaporation_rate: Pheromone evaporation rate (ρ), range [0, 1]
            q: Constant for pheromone deposit calculation
            seed: Random seed for reproducibility
        """
        super().__init__(distance_matrix)
        
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Algorithm-specific attributes
        self.pheromone: Optional[np.ndarray] = None
        self.heuristic: Optional[np.ndarray] = None
        self.tau_0: float = 0.0
        
    def _initialize(self) -> None:
        """Initialize pheromone matrix and heuristic information."""
        # Heuristic information: η_ij = 1 / d_ij (visibility)
        self.heuristic = np.zeros_like(self.distance_matrix)
        mask = self.distance_matrix > 0
        self.heuristic[mask] = 1.0 / self.distance_matrix[mask]
        
        # Initial pheromone: τ_0 = num_ants / (approximate tour length)
        # Use nearest neighbor heuristic for initial estimate
        nn_tour_length = self._nearest_neighbor_tour_length()
        self.tau_0 = self.num_ants / nn_tour_length
        
        # Initialize pheromone matrix with τ_0
        self.pheromone = np.full(
            (self.num_cities, self.num_cities),
            self.tau_0,
            dtype=np.float64
        )
        
    def _nearest_neighbor_tour_length(self) -> float:
        """
        Calculate tour length using nearest neighbor heuristic for initialization.
        
        Returns:
            Approximate tour length
        """
        unvisited = set(range(self.num_cities))
        current = 0
        unvisited.remove(current)
        tour_length = 0.0
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current, city])
            tour_length += self.distance_matrix[current, nearest]
            current = nearest
            unvisited.remove(current)
        
        # Return to start
        tour_length += self.distance_matrix[current, 0]
        return tour_length
    
    def solve(
        self,
        max_iterations: int = 100,
        early_stopping: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[int], float]:
        """
        Solve the TSP using Ant System algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            early_stopping: Stop if no improvement for this many iterations
            verbose: Print progress information
        
        Returns:
            Tuple of (best_tour, best_distance)
        """
        self.reset()
        self._initialize()
        
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            # Construct solutions for all ants
            iteration_tours = []
            iteration_distances = []
            
            for ant in range(self.num_ants):
                tour = self._construct_solution()
                distance = self.calculate_tour_distance(tour)
                iteration_tours.append(tour)
                iteration_distances.append(distance)
                
                # Update best solution if improved
                if self._update_best_solution(tour, distance):
                    iterations_without_improvement = 0
                    if verbose:
                        print(f"Iteration {iteration + 1}: New best = {distance:.2f}")
                else:
                    iterations_without_improvement += 1
            
            # Update pheromone trails
            self._update_pheromones(iteration_tours, iteration_distances)
            
            # Record history
            self.history['best_distances'].append(self.best_distance)
            self.history['iteration_distances'].append(iteration_distances)
            self.history['iterations'].append(iteration + 1)
            
            # Early stopping
            if early_stopping and iterations_without_improvement >= early_stopping:
                if verbose:
                    print(f"Early stopping at iteration {iteration + 1}")
                break
        
        return self.get_best_solution()
    
    def _construct_solution(self) -> List[int]:
        """
        Construct a solution (tour) for one ant using the probabilistic rule.
        
        Returns:
            List of city indices representing a complete tour
        """
        tour = []
        unvisited = set(range(self.num_cities))
        
        # Start from a random city
        current_city = self.rng.integers(0, self.num_cities)
        tour.append(current_city)
        unvisited.remove(current_city)
        
        # Construct the rest of the tour
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return tour
    
    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        """
        Select the next city to visit using the probabilistic transition rule.
        
        Args:
            current_city: Current city index
            unvisited: Set of unvisited city indices
        
        Returns:
            Next city to visit
        """
        unvisited_list = list(unvisited)
        
        # Calculate probabilities: p_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        pheromone_values = self.pheromone[current_city, unvisited_list]
        heuristic_values = self.heuristic[current_city, unvisited_list]
        
        # Avoid numerical issues
        pheromone_values = np.maximum(pheromone_values, 1e-10)
        heuristic_values = np.maximum(heuristic_values, 1e-10)
        
        # Calculate attractiveness
        attractiveness = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities = attractiveness / attractiveness.sum()
        
        # Select next city according to probabilities
        next_city_idx = self.rng.choice(len(unvisited_list), p=probabilities)
        return unvisited_list[next_city_idx]
    
    def _update_pheromones(self, tours: List[List[int]], distances: List[float]) -> None:
        """
        Update pheromone trails using AS pheromone update rule.
        
        Args:
            tours: List of tours constructed by all ants
            distances: List of tour distances
        """
        # Evaporation: τ_ij ← (1 - ρ) * τ_ij
        self.pheromone *= (1.0 - self.evaporation_rate)
        
        # Pheromone deposit: Each ant deposits pheromone on its tour
        for tour, distance in zip(tours, distances):
            deposit = self.q / distance
            
            for i in range(len(tour)):
                city_a = tour[i]
                city_b = tour[(i + 1) % len(tour)]
                self.pheromone[city_a, city_b] += deposit
                self.pheromone[city_b, city_a] += deposit
    
    def get_pheromone_matrix(self) -> np.ndarray:
        """
        Get the current pheromone matrix.
        
        Returns:
            Pheromone matrix
        """
        return self.pheromone.copy()
    
    def get_parameters(self) -> dict:
        """
        Get the algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'algorithm': 'Ant System (AS)',
            'num_ants': self.num_ants,
            'alpha': self.alpha,
            'beta': self.beta,
            'evaporation_rate': self.evaporation_rate,
            'q': self.q,
            'tau_0': self.tau_0
        }
    
    def __repr__(self) -> str:
        return (
            f"AntSystem(num_cities={self.num_cities}, num_ants={self.num_ants}, "
            f"alpha={self.alpha}, beta={self.beta}, rho={self.evaporation_rate}, "
            f"best_distance={self.best_distance:.2f})"
        )
