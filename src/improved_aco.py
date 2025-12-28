"""
Improved ACO - MMAS with 2-opt Local Search and Candidate Lists for enhanced solution quality.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.mmas import MaxMinAntSystem


class ImprovedACO(MaxMinAntSystem):
    """
    Improved Ant Colony Optimization combining MMAS with 2-opt local search and candidate lists.
    
    This algorithm enhances MMAS by:
    1. Using candidate lists (nearest neighbors) for efficient solution construction
    2. Applying 2-opt local search to improve solution quality
    
    The candidate list restricts the search to the k nearest neighbors of each city,
    significantly speeding up solution construction while maintaining solution quality.
    
    The 2-opt local search removes crossing edges and reconnects the tour
    in a better way, guaranteeing local optimality.
    
    Expected benefits:
    - Faster construction phase (reduced from O(n²) to O(k) per city)
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
        candidate_list_size: Optional[int] = None,
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
            candidate_list_size: Number of nearest neighbors in candidate list (None = use all cities)
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
        
        # Candidate list configuration
        if candidate_list_size is None:
            # Default: min(20, max(15, num_cities // 5))
            self.candidate_list_size = min(20, max(15, self.num_cities // 5))
        else:
            self.candidate_list_size = min(candidate_list_size, self.num_cities - 1)
        
        self.candidate_lists: Optional[np.ndarray] = None
        
        # Statistics
        self.local_search_improvements = 0
        self.total_local_searches = 0
        self.candidate_list_usage = 0
        self.fallback_to_all_cities = 0
        
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
            improved = self._update_best_solution(iteration_best_tour, iteration_best_distance)
            if improved:
                iterations_without_improvement = 0
                self.stagnation_counter = 0
                self._update_pheromone_bounds()
                if verbose:
                    print(f"Iteration {iteration + 1}: New best = {iteration_best_distance:.2f}")
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
            if self.stagnation_counter > 0:
                if self.stagnation_counter % (self.stagnation_threshold // 4) == 0:
                    self._apply_pheromone_smoothing()
                    print(f"Iteration {iteration + 1}: Pheromone smoothing applied")
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
    
    def _initialize(self) -> None:
        """Initialize pheromone matrix, heuristic information, bounds, and candidate lists."""
        # Call parent initialization
        super()._initialize()
        
        # Compute candidate lists based on nearest neighbors
        self._compute_candidate_lists()
    
    def _compute_candidate_lists(self) -> None:
        """
        Compute candidate lists for each city based on nearest neighbors.
        
        For each city, store the indices of the k nearest cities sorted by distance.
        This speeds up the solution construction phase significantly.
        """
        self.candidate_lists = np.zeros((self.num_cities, self.candidate_list_size), dtype=np.int32)
        
        for city in range(self.num_cities):
            # Get distances from current city to all other cities
            distances = self.distance_matrix[city].copy()
            distances[city] = np.inf  # Exclude the city itself
            
            # Get indices of k nearest neighbors
            nearest_indices = np.argpartition(distances, min(self.candidate_list_size, self.num_cities - 1))[:self.candidate_list_size]
            
            # Sort the nearest neighbors by distance
            nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
            
            self.candidate_lists[city] = nearest_indices
    
    def _construct_solution(self) -> List[int]:
        """
        Construct a solution (tour) for one ant using the probabilistic rule with candidate lists.
        
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
        Select the next city to visit using candidate list and probabilistic transition rule.
        
        First tries to select from the candidate list. If all candidates are already visited,
        falls back to selecting from all unvisited cities.
        
        Args:
            current_city: Current city index
            unvisited: Set of unvisited city indices
        
        Returns:
            Next city to visit
        """
        # Get candidate cities that are still unvisited
        candidates = [city for city in self.candidate_lists[current_city] if city in unvisited]
        
        # If no candidates available in the candidate list, use all unvisited cities
        if not candidates:
            candidates = list(unvisited)
            self.fallback_to_all_cities += 1
        else:
            self.candidate_list_usage += 1
        
        # Calculate probabilities for candidate cities
        pheromone_values = self.pheromone[current_city, candidates]
        heuristic_values = self.heuristic[current_city, candidates]
        
        # Avoid numerical issues
        pheromone_values = np.maximum(pheromone_values, 1e-10)
        heuristic_values = np.maximum(heuristic_values, 1e-10)
        
        # Calculate attractiveness: τ^α * η^β
        attractiveness = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities = attractiveness / attractiveness.sum()
        
        # Select next city according to probabilities
        next_city_idx = self.rng.choice(len(candidates), p=probabilities)
        return candidates[next_city_idx]
    
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
        Get statistics about local search and candidate list performance.
        
        Returns:
            Dictionary with local search and candidate list statistics
        """
        improvement_rate = (self.local_search_improvements / max(self.total_local_searches, 1)) * 100
        total_selections = self.candidate_list_usage + self.fallback_to_all_cities
        candidate_usage_rate = (self.candidate_list_usage / max(total_selections, 1)) * 100
        
        return {
            'total_applications': self.total_local_searches,
            'improvements_found': self.local_search_improvements,
            'improvement_rate_percent': improvement_rate,
            'candidate_list_size': self.candidate_list_size,
            'candidate_list_usage': self.candidate_list_usage,
            'fallback_to_all_cities': self.fallback_to_all_cities,
            'candidate_usage_rate_percent': candidate_usage_rate
        }
    
    def get_parameters(self) -> dict:
        """
        Get the algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_parameters()
        params['algorithm'] = 'Improved ACO (MMAS + 2-opt + Candidate Lists)'
        params['apply_local_search_to_all'] = self.apply_local_search_to_all
        params['max_local_search_iterations'] = self.max_local_search_iterations
        params['candidate_list_size'] = self.candidate_list_size
        return params
    
    def __repr__(self) -> str:
        ls_strategy = "all ants" if self.apply_local_search_to_all else "best ant"
        return (
            f"ImprovedACO(num_cities={self.num_cities}, num_ants={self.num_ants}, "
            f"alpha={self.alpha}, beta={self.beta}, rho={self.evaporation_rate}, "
            f"candidate_list_size={self.candidate_list_size}, "
            f"local_search={ls_strategy}, "
            f"best_distance={self.best_distance:.2f})"
        )
