"""
Max-Min Ant System (MMAS) - An improved ACO algorithm with pheromone limits.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.tsp_algorithm import TSPAlgorithm


class MaxMinAntSystem(TSPAlgorithm):
    """
    Max-Min Ant System (MMAS) algorithm for solving the Traveling Salesman Problem.
    
    MMAS improves upon AS by:
    - Limiting pheromone trails to [tau_min, tau_max]
    - Only allowing the best ant to update pheromones
    - Reinitializing pheromones when stagnation is detected
    
    Reference: Stützle & Hoos (2000)
    """
    
    def __init__(
        self,
        distance_matrix: np.ndarray,
        num_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.02,
        use_iteration_best: bool = True,
        pbest: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Initialize the Max-Min Ant System algorithm.
        
        Args:
            distance_matrix: Square matrix of distances between cities
            num_ants: Number of ants in the colony
            alpha: Pheromone importance factor (α)
            beta: Heuristic information importance factor (β)
            evaporation_rate: Pheromone evaporation rate (ρ), range [0, 1]
            use_iteration_best: If True, use iteration-best; else use global-best
            pbest: Probability for calculating tau_max (typically 0.05)
            seed: Random seed for reproducibility
        """
        super().__init__(distance_matrix)
        
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.use_iteration_best = use_iteration_best
        self.pbest = pbest
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Algorithm-specific attributes
        self.pheromone: Optional[np.ndarray] = None
        self.heuristic: Optional[np.ndarray] = None
        self.tau_min: float = 0.0
        self.tau_max: float = 0.0
        
        # Stagnation detection
        self.stagnation_counter: int = 0
        self.stagnation_threshold: int = 0
        
    def _initialize(self) -> None:
        """Initialize pheromone matrix, heuristic information, and bounds."""
        # Heuristic information: η_ij = 1 / d_ij (visibility)
        self.heuristic = np.zeros_like(self.distance_matrix)
        mask = self.distance_matrix > 0
        self.heuristic[mask] = 1.0 / self.distance_matrix[mask]
        
        # Get initial solution using nearest neighbor
        nn_tour, nn_distance = self._nearest_neighbor_solution()
        self._update_best_solution(nn_tour, nn_distance)
        
        # Initialize pheromone bounds
        self._update_pheromone_bounds()
        
        # Initialize pheromone matrix with τ_max
        self.pheromone = np.full(
            (self.num_cities, self.num_cities),
            self.tau_max,
            dtype=np.float64
        )
        
        # Stagnation threshold (reinitialize if no improvement for this many iterations)
        self.stagnation_threshold = max(25, self.num_cities // 2)
        self.stagnation_counter = 0
        
    def _nearest_neighbor_solution(self) -> Tuple[List[int], float]:
        """
        Construct a solution using nearest neighbor heuristic.
        
        Returns:
            Tuple of (tour, tour_distance)
        """
        tour = []
        unvisited = set(range(self.num_cities))
        
        current = 0
        tour.append(current)
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        distance = self.calculate_tour_distance(tour)
        return tour, distance
    
    def _update_pheromone_bounds(self) -> None:
        """
        Update τ_max and τ_min based on the current best solution.
        
        τ_max = 1 / (ρ * L_best)
        τ_min = τ_max * (1 - p_best^(1/n)) / ((avg - 1) * p_best^(1/n))
        """
        # tau_max
        self.tau_max = 1.0 / (self.evaporation_rate * self.best_distance)
        
        # tau_min (simplified formula)
        avg = self.num_cities / 2.0
        p_dec = self.pbest ** (1.0 / self.num_cities)
        self.tau_min = self.tau_max * (1.0 - p_dec) / ((avg - 1.0) * p_dec)
        
        # Ensure tau_min is positive and less than tau_max
        self.tau_min = max(self.tau_min, 1e-10)
        self.tau_min = min(self.tau_min, self.tau_max / 2.0)
    
    def solve(
        self,
        max_iterations: int = 100,
        early_stopping: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[int], float]:
        """
        Solve the TSP using Max-Min Ant System algorithm.
        
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
        iteration_best_tour = None
        iteration_best_distance = float('inf')
        
        for iteration in range(max_iterations):
            # Construct solutions for all ants
            iteration_tours = []
            iteration_distances = []
            
            iteration_best_tour = None
            iteration_best_distance = float('inf')
            
            for ant in range(self.num_ants):
                tour = self._construct_solution()
                distance = self.calculate_tour_distance(tour)
                iteration_tours.append(tour)
                iteration_distances.append(distance)
                
                # Track iteration-best
                if distance < iteration_best_distance:
                    iteration_best_distance = distance
                    iteration_best_tour = tour
                
                # Update global best solution
                improved = self._update_best_solution(tour, distance)
                if improved:
                    iterations_without_improvement = 0
                    self.stagnation_counter = 0
                    self._update_pheromone_bounds()
                    if verbose:
                        print(f"Iteration {iteration + 1}: New best = {distance:.2f}")
                else:
                    iterations_without_improvement += 1
                    self.stagnation_counter += 1
            
            # Choose which tour to use for pheromone update
            if self.use_iteration_best:
                update_tour = iteration_best_tour
                update_distance = iteration_best_distance
            else:
                update_tour = self.best_tour
                update_distance = self.best_distance
            
            # Update pheromone trails
            self._update_pheromones(update_tour, update_distance)
            
            # Check for stagnation and reinitialize if needed
            if self.stagnation_counter >= self.stagnation_threshold:
                self._reinitialize_pheromones()
                self.stagnation_counter = 0
                if verbose:
                    print(f"Iteration {iteration + 1}: Pheromone reinitialization")
            
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
    
    def _update_pheromones(self, tour: List[int], distance: float) -> None:
        """
        Update pheromone trails using MMAS pheromone update rule.
        Only the best ant deposits pheromone.
        
        Args:
            tour: Best tour (iteration-best or global-best)
            distance: Distance of the best tour
        """
        # Evaporation: τ_ij ← (1 - ρ) * τ_ij
        self.pheromone *= (1.0 - self.evaporation_rate)
        
        # Pheromone deposit: Only best ant deposits
        deposit = 1.0 / distance
        
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            self.pheromone[city_a, city_b] += deposit
            self.pheromone[city_b, city_a] += deposit
        
        # Apply pheromone limits [tau_min, tau_max]
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
    
    def _reinitialize_pheromones(self) -> None:
        """
        Reinitialize pheromone matrix when stagnation is detected.
        All pheromone values are reset to tau_max.
        """
        self._update_pheromone_bounds()
        self.pheromone.fill(self.tau_max)
    
    def _apply_pheromone_smoothing(self, smoothing_factor: float = 0.1) -> None:
        """
        Apply pheromone smoothing to avoid premature convergence.
        
        Args:
            smoothing_factor: Smoothing intensity
        """
        mean_pheromone = np.mean(self.pheromone)
        self.pheromone = (1 - smoothing_factor) * self.pheromone + smoothing_factor * mean_pheromone
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
    
    def get_pheromone_matrix(self) -> np.ndarray:
        """
        Get the current pheromone matrix.
        
        Returns:
            Pheromone matrix
        """
        return self.pheromone.copy()
    
    def get_pheromone_bounds(self) -> Tuple[float, float]:
        """
        Get the current pheromone bounds.
        
        Returns:
            Tuple of (tau_min, tau_max)
        """
        return self.tau_min, self.tau_max
    
    def get_parameters(self) -> dict:
        """
        Get the algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'algorithm': 'Max-Min Ant System (MMAS)',
            'num_ants': self.num_ants,
            'alpha': self.alpha,
            'beta': self.beta,
            'evaporation_rate': self.evaporation_rate,
            'use_iteration_best': self.use_iteration_best,
            'pbest': self.pbest,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'stagnation_threshold': self.stagnation_threshold
        }
    
    def __repr__(self) -> str:
        return (
            f"MaxMinAntSystem(num_cities={self.num_cities}, num_ants={self.num_ants}, "
            f"alpha={self.alpha}, beta={self.beta}, rho={self.evaporation_rate}, "
            f"tau_range=[{self.tau_min:.6f}, {self.tau_max:.6f}], "
            f"best_distance={self.best_distance:.2f})"
        )
