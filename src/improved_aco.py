"""
Improved ACO - MMAS with 2-opt Local Search for enhanced solution quality.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.mmas import MaxMinAntSystem


class ImprovedACO(MaxMinAntSystem):
    """
    Improved Ant Colony Optimization combining MMAS with 2-opt local search.
    
    This algorithm enhances MMAS by applying 2-opt local search to improve
    solution quality. The local search can be applied to:
    - Only the best ant's tour (faster, less computational cost)
    - All ants' tours (slower, potentially better exploration)
    
    The 2-opt local search removes crossing edges and reconnects the tour
    in a better way, guaranteeing local optimality.
    
    Expected benefits:
    - Better convergence speed
    - Shorter final path lengths
    - Higher quality solutions
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
        apply_local_search_to_all: bool = False,
        max_local_search_iterations: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize the Improved ACO algorithm.
        
        Args:
            distance_matrix: Square matrix of distances between cities
            num_ants: Number of ants in the colony
            alpha: Pheromone importance factor (α)
            beta: Heuristic information importance factor (β)
            evaporation_rate: Pheromone evaporation rate (ρ)
            use_iteration_best: If True, use iteration-best; else use global-best
            pbest: Probability for calculating tau_max
            apply_local_search_to_all: If True, apply 2-opt to all ants; else only to best
            max_local_search_iterations: Maximum iterations for 2-opt (early stopping)
            seed: Random seed for reproducibility
        """
        super().__init__(
            distance_matrix=distance_matrix,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            use_iteration_best=use_iteration_best,
            pbest=pbest,
            seed=seed
        )
        
        self.apply_local_search_to_all = apply_local_search_to_all
        self.max_local_search_iterations = max_local_search_iterations
        
        # Statistics
        self.local_search_improvements = 0
        self.total_local_searches = 0
        
    def solve(
        self,
        max_iterations: int = 100,
        early_stopping: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[int], float]:
        """
        Solve the TSP using Improved ACO (MMAS + 2-opt local search).
        
        Args:
            max_iterations: Maximum number of iterations
            early_stopping: Stop if no improvement for this many iterations
            verbose: Print progress information
        
        Returns:
            Tuple of (best_tour, best_distance)
        """
        self.reset()
        self._initialize()
        
        self.local_search_improvements = 0
        self.total_local_searches = 0
        
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
                
                # Apply local search based on strategy
                if self.apply_local_search_to_all:
                    # Apply 2-opt to all ants
                    tour, distance = self._apply_2opt(tour, distance)
                
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
            
            # If not applying to all, apply 2-opt to iteration-best only
            if not self.apply_local_search_to_all:
                iteration_best_tour, iteration_best_distance = self._apply_2opt(
                    iteration_best_tour, iteration_best_distance
                )
                
                # Check if local search improved the best solution
                if self._update_best_solution(iteration_best_tour, iteration_best_distance):
                    iterations_without_improvement = 0
                    self.stagnation_counter = 0
                    self._update_pheromone_bounds()
                    if verbose:
                        print(f"Iteration {iteration + 1}: Local search improved to {iteration_best_distance:.2f}")
            
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
        
        if verbose:
            improvement_rate = (self.local_search_improvements / max(self.total_local_searches, 1)) * 100
            print(f"\nLocal Search Statistics:")
            print(f"  Total applications: {self.total_local_searches}")
            print(f"  Improvements found: {self.local_search_improvements}")
            print(f"  Improvement rate: {improvement_rate:.1f}%")
        
        return self.get_best_solution()
    
    def _apply_2opt(self, tour: List[int], current_distance: float) -> Tuple[List[int], float]:
        """
        Apply 2-opt local search to improve a tour.
        
        The 2-opt algorithm works by:
        1. Selecting two edges (i, i+1) and (j, j+1)
        2. Removing these edges
        3. Reconnecting the tour by reversing the segment between them
        4. Keeping the change if it improves the tour
        
        Args:
            tour: Current tour to improve
            current_distance: Current tour distance
        
        Returns:
            Tuple of (improved_tour, improved_distance)
        """
        self.total_local_searches += 1
        improved_tour = tour.copy()
        improved_distance = current_distance
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_local_search_iterations:
            improved = False
            iterations += 1
            
            for i in range(self.num_cities - 1):
                for j in range(i + 2, self.num_cities):
                    # Avoid adjacent edges (they can't be swapped meaningfully)
                    if j == self.num_cities - 1 and i == 0:
                        continue
                    
                    # Calculate the change in distance from this 2-opt move
                    # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                    city_i = improved_tour[i]
                    city_i_next = improved_tour[i + 1]
                    city_j = improved_tour[j]
                    city_j_next = improved_tour[(j + 1) % self.num_cities]
                    
                    # Calculate distance change
                    old_distance = (
                        self.distance_matrix[city_i, city_i_next] +
                        self.distance_matrix[city_j, city_j_next]
                    )
                    new_distance = (
                        self.distance_matrix[city_i, city_j] +
                        self.distance_matrix[city_i_next, city_j_next]
                    )
                    
                    distance_change = new_distance - old_distance
                    
                    # If improvement found, apply the 2-opt swap
                    if distance_change < -1e-9:  # Use small epsilon for floating point comparison
                        # Reverse the segment between i+1 and j
                        improved_tour[i + 1:j + 1] = improved_tour[i + 1:j + 1][::-1]
                        improved_distance += distance_change
                        improved = True
                        self.local_search_improvements += 1
                        break  # Start over with the new tour
                
                if improved:
                    break  # Start over from the beginning
        
        return improved_tour, improved_distance
    
    def _apply_2opt_optimized(self, tour: List[int], current_distance: float) -> Tuple[List[int], float]:
        """
        Optimized version of 2-opt using vectorized distance calculations.
        
        This version can be faster for large instances by reducing Python loops.
        
        Args:
            tour: Current tour to improve
            current_distance: Current tour distance
        
        Returns:
            Tuple of (improved_tour, improved_distance)
        """
        self.total_local_searches += 1
        improved_tour = np.array(tour)
        improved_distance = current_distance
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_local_search_iterations:
            improved = False
            iterations += 1
            best_i, best_j = -1, -1
            best_improvement = 0
            
            # Find the best 2-opt swap in this iteration
            for i in range(self.num_cities - 1):
                for j in range(i + 2, self.num_cities):
                    if j == self.num_cities - 1 and i == 0:
                        continue
                    
                    city_i = improved_tour[i]
                    city_i_next = improved_tour[i + 1]
                    city_j = improved_tour[j]
                    city_j_next = improved_tour[(j + 1) % self.num_cities]
                    
                    old_distance = (
                        self.distance_matrix[city_i, city_i_next] +
                        self.distance_matrix[city_j, city_j_next]
                    )
                    new_distance = (
                        self.distance_matrix[city_i, city_j] +
                        self.distance_matrix[city_i_next, city_j_next]
                    )
                    
                    improvement = old_distance - new_distance
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i, j
                        improved = True
            
            # Apply the best swap found
            if improved:
                improved_tour[best_i + 1:best_j + 1] = improved_tour[best_i + 1:best_j + 1][::-1]
                improved_distance -= best_improvement
                self.local_search_improvements += 1
        
        return improved_tour.tolist(), improved_distance
    
    def get_local_search_statistics(self) -> dict:
        """
        Get statistics about local search performance.
        
        Returns:
            Dictionary with local search statistics
        """
        improvement_rate = (self.local_search_improvements / max(self.total_local_searches, 1)) * 100
        return {
            'total_applications': self.total_local_searches,
            'improvements_found': self.local_search_improvements,
            'improvement_rate_percent': improvement_rate
        }
    
    def get_parameters(self) -> dict:
        """
        Get the algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_parameters()
        params['algorithm'] = 'Improved ACO (MMAS + 2-opt)'
        params['apply_local_search_to_all'] = self.apply_local_search_to_all
        params['max_local_search_iterations'] = self.max_local_search_iterations
        return params
    
    def __repr__(self) -> str:
        ls_strategy = "all ants" if self.apply_local_search_to_all else "best ant"
        return (
            f"ImprovedACO(num_cities={self.num_cities}, num_ants={self.num_ants}, "
            f"alpha={self.alpha}, beta={self.beta}, rho={self.evaporation_rate}, "
            f"local_search={ls_strategy}, "
            f"best_distance={self.best_distance:.2f})"
        )
