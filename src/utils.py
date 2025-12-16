"""
Utility functions for visualization and analysis of TSP solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from matplotlib.patches import FancyBboxPatch


def plot_convergence(
    history: Dict[str, List],
    title: str = "Convergence Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the convergence curve showing best distance over iterations.
    
    Args:
        history: Dictionary with 'iterations' and 'best_distances' keys
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['iterations'], history['best_distances'], 
             linewidth=2, color='#2E86AB', marker='o', markersize=3,
             markerfacecolor='#A23B72', markeredgewidth=0)
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Best Distance Found', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tour(
    coordinates: np.ndarray,
    tour: List[int],
    title: str = "Best Tour Found",
    distance: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the TSP tour showing cities and connections.
    
    Args:
        coordinates: Array of city coordinates (n, 2)
        tour: List of city indices in tour order
        title: Plot title
        distance: Total tour distance (optional, for display)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot the tour edges
    for i in range(len(tour)):
        start_idx = tour[i]
        end_idx = tour[(i + 1) % len(tour)]
        
        start = coordinates[start_idx]
        end = coordinates[end_idx]
        
        plt.plot([start[0], end[0]], [start[1], end[1]], 
                'b-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Plot the cities
    plt.scatter(coordinates[:, 0], coordinates[:, 1], 
               c='red', s=100, zorder=2, edgecolors='black', linewidth=1.5)
    
    # Highlight start city
    start_city = coordinates[tour[0]]
    plt.scatter(start_city[0], start_city[1], 
               c='green', s=200, zorder=3, marker='*', 
               edgecolors='black', linewidth=2, label='Start City')
    
    # Add city labels (optional for small instances)
    if len(tour) <= 50:
        for i, (x, y) in enumerate(coordinates):
            plt.annotate(str(i), (x, y), fontsize=8, 
                        ha='center', va='center', color='white', 
                        weight='bold', zorder=4)
    
    title_text = title
    if distance is not None:
        title_text += f"\nTotal Distance: {distance:.2f}"
    
    plt.xlabel('X Coordinate', fontsize=12, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tour plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, Dict],
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot convergence curves for multiple algorithms on the same graph.
    
    Args:
        results: Dictionary mapping algorithm names to result dictionaries
                 Each result should have 'history' key with convergence data
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for idx, (algo_name, result) in enumerate(results.items()):
        history = result['history']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(history['iterations'], history['best_distances'],
                linewidth=2, color=color, marker=marker, markersize=4,
                markevery=max(1, len(history['iterations']) // 20),
                label=algo_name, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Best Distance Found', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comprehensive_comparison(
    results: Dict[str, Dict],
    coordinates: np.ndarray,
    problem_name: str = "TSP Instance",
    save_dir: Optional[str] = None
) -> None:
    """
    Create a comprehensive comparison figure with multiple subplots.
    
    Args:
        results: Dictionary mapping algorithm names to result dictionaries
        coordinates: Array of city coordinates for tour visualization
        problem_name: Name of the problem instance
        save_dir: Directory to save the figure (optional)
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Convergence comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for idx, (algo_name, result) in enumerate(results.items()):
        history = result['history']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax1.plot(history['iterations'], history['best_distances'],
                linewidth=2, color=color, marker=marker, markersize=4,
                markevery=max(1, len(history['iterations']) // 20),
                label=algo_name, alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Best Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Performance metrics table (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    metrics_data = []
    for algo_name, result in results.items():
        metrics_data.append([
            algo_name,
            f"{result['best_distance']:.2f}",
            f"{result['execution_time']:.2f}s",
            f"{result['convergence_iteration']}"
        ])
    
    table = ax2.table(cellText=metrics_data,
                     colLabels=['Algorithm', 'Best Dist.', 'Time', 'Conv. Iter'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the data rows
    for i in range(1, len(metrics_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
    
    ax2.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Plot best tours for each algorithm (bottom row)
    for idx, (algo_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])
        
        tour = result['best_tour']
        
        # Plot tour edges
        for i in range(len(tour)):
            start_idx = tour[i]
            end_idx = tour[(i + 1) % len(tour)]
            start = coordinates[start_idx]
            end = coordinates[end_idx]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   'b-', linewidth=1, alpha=0.6)
        
        # Plot cities
        ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                  c='red', s=50, zorder=2, edgecolors='black', linewidth=1)
        
        # Highlight start
        start_city = coordinates[tour[0]]
        ax.scatter(start_city[0], start_city[1], 
                  c='green', s=100, zorder=3, marker='*', 
                  edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_title(f'{algo_name}\nDist: {result["best_distance"]:.2f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    fig.suptitle(f'Comprehensive Analysis: {problem_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{problem_name}_comprehensive.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to: {save_path}")
    
    plt.show()


def print_statistics(
    algo_name: str,
    best_distance: float,
    convergence_iteration: int,
    execution_time: float,
    total_iterations: int,
    additional_stats: Optional[Dict] = None
) -> None:
    """
    Print formatted statistics for an algorithm run.
    
    Args:
        algo_name: Name of the algorithm
        best_distance: Best distance found
        convergence_iteration: Iteration where best solution was found
        execution_time: Total execution time in seconds
        total_iterations: Total number of iterations run
        additional_stats: Optional dictionary of additional statistics
    """
    print("\n" + "=" * 70)
    print(f"  {algo_name} - Results")
    print("=" * 70)
    print(f"  Best Distance:          {best_distance:.2f}")
    print(f"  Convergence Iteration:  {convergence_iteration} / {total_iterations}")
    print(f"  Execution Time:         {execution_time:.2f} seconds")
    
    if additional_stats:
        print("\n  Additional Statistics:")
        for key, value in additional_stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
    
    print("=" * 70)


def calculate_metrics(history: Dict[str, List]) -> Dict[str, float]:
    """
    Calculate various performance metrics from optimization history.
    
    Args:
        history: Dictionary with optimization history
    
    Returns:
        Dictionary of calculated metrics
    """
    best_distances = history['best_distances']
    
    metrics = {
        'final_best': best_distances[-1],
        'initial_best': best_distances[0],
        'improvement': best_distances[0] - best_distances[-1],
        'improvement_percent': ((best_distances[0] - best_distances[-1]) / best_distances[0]) * 100,
        'convergence_iteration': np.argmin(best_distances) + 1,
        'average_improvement_per_iter': (best_distances[0] - best_distances[-1]) / len(best_distances)
    }
    
    return metrics


def save_tour_to_file(
    tour: List[int],
    distance: float,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save tour to a text file.
    
    Args:
        tour: List of city indices
        distance: Total tour distance
        filepath: Path to save the file
        metadata: Optional metadata dictionary
    """
    with open(filepath, 'w') as f:
        if metadata:
            f.write("METADATA\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write(f"TOUR DISTANCE: {distance:.2f}\n")
        f.write(f"TOUR LENGTH: {len(tour)}\n\n")
        f.write("TOUR:\n")
        for i, city in enumerate(tour):
            f.write(f"{i + 1}: {city}\n")
    
    print(f"Tour saved to: {filepath}")
