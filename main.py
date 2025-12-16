"""
Main driver script for TSP solver using Ant Colony Optimization.

This script provides a command-line interface to run and compare different
ACO algorithms (AS, MMAS, ImprovedACO) on TSPLIB benchmark instances.
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import components
from src.data_loader import DataLoader
from src.ant_system import AntSystem
from src.mmas import MaxMinAntSystem
from src.improved_aco import ImprovedACO
from src.utils import (
    plot_convergence, plot_tour, plot_comparison,
    plot_comprehensive_comparison, print_statistics,
    calculate_metrics, save_tour_to_file
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Solve TSP using Ant Colony Optimization algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Ant System on berlin52
  python main.py data/berlin52.tsp --algorithm AS --ants 20 --iterations 100
  
  # Run MMAS with custom parameters
  python main.py data/berlin52.tsp --algorithm MMAS --ants 30 --alpha 1.0 --beta 2.5
  
  # Compare all algorithms
  python main.py data/berlin52.tsp --compare --iterations 100 --verbose
  
  # Run ImprovedACO and save results
  python main.py data/eil76.tsp --algorithm Improved --save-results results/
        """
    )
    
    # Required arguments
    parser.add_argument('file', type=str, 
                       help='Path to TSPLIB format .tsp file')
    
    # Algorithm selection
    parser.add_argument('--algorithm', '-a', type=str, 
                       choices=['AS', 'MMAS', 'Improved'], default='MMAS',
                       help='Algorithm to use (default: MMAS)')
    
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare all three algorithms')
    
    # Algorithm parameters
    parser.add_argument('--ants', type=int, default=20,
                       help='Number of ants (default: 20)')
    
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Maximum number of iterations (default: 100)')
    
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Pheromone importance factor (default: 1.0)')
    
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Heuristic importance factor (default: 2.0)')
    
    parser.add_argument('--rho', type=float, default=None,
                       help='Evaporation rate (default: 0.5 for AS, 0.02 for MMAS/Improved)')
    
    parser.add_argument('--q', type=float, default=100.0,
                       help='Pheromone deposit constant for AS (default: 100.0)')
    
    # Execution options
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of independent runs (default: 1)')
    
    parser.add_argument('--early-stopping', type=int, default=None,
                       help='Early stopping after N iterations without improvement')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')
    
    parser.add_argument('--save-results', type=str, default=None,
                       help='Directory to save results (plots and tour files)')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not display plots (useful for batch processing)')
    
    return parser.parse_args()


def create_algorithm(
    algo_name: str,
    distance_matrix: np.ndarray,
    num_ants: int,
    alpha: float,
    beta: float,
    rho: Optional[float],
    q: float,
    seed: Optional[int]
):
    """
    Factory function to create algorithm instances.
    
    Args:
        algo_name: Name of algorithm ('AS', 'MMAS', 'Improved')
        distance_matrix: Distance matrix
        num_ants: Number of ants
        alpha: Pheromone importance
        beta: Heuristic importance
        rho: Evaporation rate (None for default)
        q: Pheromone deposit constant
        seed: Random seed
    
    Returns:
        Algorithm instance
    """
    if algo_name == 'AS':
        evap_rate = rho if rho is not None else 0.5
        return AntSystem(
            distance_matrix=distance_matrix,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evap_rate,
            q=q,
            seed=seed
        )
    
    elif algo_name == 'MMAS':
        evap_rate = rho if rho is not None else 0.02
        return MaxMinAntSystem(
            distance_matrix=distance_matrix,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evap_rate,
            seed=seed
        )
    
    elif algo_name == 'Improved':
        evap_rate = rho if rho is not None else 0.02
        return ImprovedACO(
            distance_matrix=distance_matrix,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evap_rate,
            apply_local_search_to_all=False,
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_algorithm(
    algorithm,
    algo_name: str,
    max_iterations: int,
    early_stopping: Optional[int],
    verbose: bool
) -> Dict:
    """
    Run a single algorithm and collect results.
    
    Args:
        algorithm: Algorithm instance
        algo_name: Algorithm name for display
        max_iterations: Maximum iterations
        early_stopping: Early stopping threshold
        verbose: Verbose output flag
    
    Returns:
        Dictionary with results
    """
    if verbose:
        print(f"\nRunning {algo_name}...")
        print(f"  Parameters: {algorithm.get_parameters()}")
    
    start_time = time.time()
    best_tour, best_distance = algorithm.solve(
        max_iterations=max_iterations,
        early_stopping=early_stopping,
        verbose=verbose
    )
    execution_time = time.time() - start_time
    
    history = algorithm.get_history()
    metrics = calculate_metrics(history)
    
    result = {
        'algorithm': algo_name,
        'best_tour': best_tour,
        'best_distance': best_distance,
        'execution_time': execution_time,
        'convergence_iteration': metrics['convergence_iteration'],
        'history': history,
        'metrics': metrics,
        'parameters': algorithm.get_parameters()
    }
    
    # Add algorithm-specific stats
    if hasattr(algorithm, 'get_local_search_statistics'):
        result['local_search_stats'] = algorithm.get_local_search_statistics()
    
    if hasattr(algorithm, 'get_pheromone_bounds'):
        result['pheromone_bounds'] = algorithm.get_pheromone_bounds()
    
    return result


def run_multiple_times(
    algo_name: str,
    distance_matrix: np.ndarray,
    num_runs: int,
    **kwargs
) -> Dict:
    """
    Run algorithm multiple times and aggregate results.
    
    Args:
        algo_name: Algorithm name
        distance_matrix: Distance matrix
        num_runs: Number of independent runs
        **kwargs: Algorithm parameters
    
    Returns:
        Dictionary with aggregated results
    """
    all_distances = []
    all_times = []
    all_convergence_iters = []
    best_result = None
    best_overall_distance = float('inf')
    
    for run in range(num_runs):
        seed = kwargs.get('seed', None)
        if seed is not None:
            seed = seed + run  # Different seed for each run
        
        algorithm = create_algorithm(
            algo_name=algo_name,
            distance_matrix=distance_matrix,
            num_ants=kwargs['num_ants'],
            alpha=kwargs['alpha'],
            beta=kwargs['beta'],
            rho=kwargs['rho'],
            q=kwargs['q'],
            seed=seed
        )
        
        result = run_algorithm(
            algorithm=algorithm,
            algo_name=f"{algo_name} (Run {run + 1}/{num_runs})",
            max_iterations=kwargs['max_iterations'],
            early_stopping=kwargs['early_stopping'],
            verbose=False  # Suppress verbose for multiple runs
        )
        
        all_distances.append(result['best_distance'])
        all_times.append(result['execution_time'])
        all_convergence_iters.append(result['convergence_iteration'])
        
        if result['best_distance'] < best_overall_distance:
            best_overall_distance = result['best_distance']
            best_result = result
            best_result['algorithm'] = algo_name  # Remove run number
    
    # Add aggregated statistics
    best_result['multi_run_stats'] = {
        'num_runs': num_runs,
        'mean_distance': np.mean(all_distances),
        'std_distance': np.std(all_distances),
        'min_distance': np.min(all_distances),
        'max_distance': np.max(all_distances),
        'mean_time': np.mean(all_times),
        'mean_convergence_iter': np.mean(all_convergence_iters)
    }
    
    return best_result


def save_results(
    result: Dict,
    coordinates: np.ndarray,
    problem_name: str,
    save_dir: str,
    show_plots: bool = True
) -> None:
    """
    Save algorithm results to files and generate plots.
    
    Args:
        result: Result dictionary
        coordinates: City coordinates
        problem_name: Problem instance name
        save_dir: Directory to save results
        show_plots: Whether to show plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    algo_name = result['algorithm']
    safe_algo_name = algo_name.replace(' ', '_')
    
    # Save tour to file
    tour_file = os.path.join(save_dir, f'{problem_name}_{safe_algo_name}_tour.txt')
    save_tour_to_file(
        tour=result['best_tour'],
        distance=result['best_distance'],
        filepath=tour_file,
        metadata={
            'Algorithm': algo_name,
            'Problem': problem_name,
            'Execution Time': f"{result['execution_time']:.2f}s",
            'Convergence Iteration': result['convergence_iteration'],
            **result['parameters']
        }
    )
    
    # Plot and save convergence curve
    conv_file = os.path.join(save_dir, f'{problem_name}_{safe_algo_name}_convergence.png')
    plot_convergence(
        history=result['history'],
        title=f"{algo_name} - Convergence on {problem_name}",
        save_path=conv_file,
        show=show_plots
    )
    
    # Plot and save tour
    tour_plot_file = os.path.join(save_dir, f'{problem_name}_{safe_algo_name}_tour.png')
    plot_tour(
        coordinates=coordinates,
        tour=result['best_tour'],
        title=f"{algo_name} - Best Tour",
        distance=result['best_distance'],
        save_path=tour_plot_file,
        show=show_plots
    )


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load TSP instance
    print("\n" + "=" * 70)
    print("  TSP Solver using Ant Colony Optimization")
    print("=" * 70)
    print(f"\nLoading problem instance: {args.file}")
    
    try:
        loader = DataLoader(args.file)
        metadata = loader.get_metadata()
        distance_matrix = loader.get_distance_matrix()
        coordinates = loader.get_coordinates()
        
        print(f"  Problem: {metadata['name']}")
        print(f"  Dimension: {metadata['dimension']} cities")
        print(f"  Edge Weight Type: {metadata['edge_weight_type']}")
    except Exception as e:
        print(f"\nError loading file: {e}")
        sys.exit(1)
    
    # Prepare parameters
    algo_params = {
        'num_ants': args.ants,
        'alpha': args.alpha,
        'beta': args.beta,
        'rho': args.rho,
        'q': args.q,
        'max_iterations': args.iterations,
        'early_stopping': args.early_stopping,
        'seed': args.seed
    }
    
    show_plots = not args.no_plot
    
    # Run algorithms
    if args.compare:
        # Comparison mode - run all three algorithms
        print("\n" + "=" * 70)
        print("  COMPARISON MODE - Running all algorithms")
        print("=" * 70)
        
        algorithms = ['AS', 'MMAS', 'Improved']
        results = {}
        
        for algo in algorithms:
            if args.runs > 1:
                result = run_multiple_times(
                    algo_name=algo,
                    distance_matrix=distance_matrix,
                    num_runs=args.runs,
                    **algo_params
                )
            else:
                algorithm = create_algorithm(
                    algo_name=algo,
                    distance_matrix=distance_matrix,
                    **algo_params
                )
                result = run_algorithm(
                    algorithm=algorithm,
                    algo_name=algo,
                    max_iterations=args.iterations,
                    early_stopping=args.early_stopping,
                    verbose=args.verbose
                )
            
            results[algo] = result
            
            # Print statistics
            additional_stats = {}
            if 'multi_run_stats' in result:
                additional_stats = result['multi_run_stats']
            elif 'local_search_stats' in result:
                additional_stats = result['local_search_stats']
            
            print_statistics(
                algo_name=algo,
                best_distance=result['best_distance'],
                convergence_iteration=result['convergence_iteration'],
                execution_time=result['execution_time'],
                total_iterations=args.iterations,
                additional_stats=additional_stats
            )
        
        # Create comprehensive comparison
        print("\nGenerating comparison visualizations...")
        
        if args.save_results:
            plot_comprehensive_comparison(
                results=results,
                coordinates=coordinates,
                problem_name=metadata['name'],
                save_dir=args.save_results
            )
            
            # Also save individual results
            for algo, result in results.items():
                save_results(
                    result=result,
                    coordinates=coordinates,
                    problem_name=metadata['name'],
                    save_dir=args.save_results,
                    show_plots=False
                )
        else:
            plot_comprehensive_comparison(
                results=results,
                coordinates=coordinates,
                problem_name=metadata['name'],
                save_dir=None
            )
        
        # Print comparison summary
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Algorithm':<15} {'Best Dist.':<12} {'Time (s)':<10} {'Conv. Iter':<12}")
        print("-" * 70)
        for algo, result in results.items():
            print(f"{algo:<15} {result['best_distance']:<12.2f} "
                  f"{result['execution_time']:<10.2f} {result['convergence_iteration']:<12}")
        print("=" * 70)
        
    else:
        # Single algorithm mode
        if args.runs > 1:
            result = run_multiple_times(
                algo_name=args.algorithm,
                distance_matrix=distance_matrix,
                num_runs=args.runs,
                **algo_params
            )
        else:
            algorithm = create_algorithm(
                algo_name=args.algorithm,
                distance_matrix=distance_matrix,
                **algo_params
            )
            result = run_algorithm(
                algorithm=algorithm,
                algo_name=args.algorithm,
                max_iterations=args.iterations,
                early_stopping=args.early_stopping,
                verbose=args.verbose
            )
        
        # Print statistics
        additional_stats = {}
        if 'multi_run_stats' in result:
            additional_stats = result['multi_run_stats']
        elif 'local_search_stats' in result:
            additional_stats = result['local_search_stats']
        
        print_statistics(
            algo_name=args.algorithm,
            best_distance=result['best_distance'],
            convergence_iteration=result['convergence_iteration'],
            execution_time=result['execution_time'],
            total_iterations=args.iterations,
            additional_stats=additional_stats
        )
        
        # Save or display results
        if args.save_results:
            save_results(
                result=result,
                coordinates=coordinates,
                problem_name=metadata['name'],
                save_dir=args.save_results,
                show_plots=show_plots
            )
        else:
            plot_convergence(
                history=result['history'],
                title=f"{args.algorithm} - Convergence on {metadata['name']}",
                show=show_plots
            )
            plot_tour(
                coordinates=coordinates,
                tour=result['best_tour'],
                title=f"{args.algorithm} - Best Tour",
                distance=result['best_distance'],
                show=show_plots
            )
    
    print("\n" + "=" * 70)
    print("  Execution Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
