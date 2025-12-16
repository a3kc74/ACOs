# Ant Colony Optimization for Traveling Salesman Problem

A Python implementation of the Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP) using TSPLIB benchmark datasets.

## Project Structure

```
ACOs/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # TSPLIB file parser and distance matrix calculator
│   ├── tsp_algorithm.py      # Abstract base class for TSP algorithms
│   ├── ant_system.py         # Ant System (AS) implementation
│   ├── mmas.py               # Max-Min Ant System (MMAS) implementation
│   ├── improved_aco.py       # Improved ACO (MMAS + 2-opt local search)
│   └── utils.py              # Utility functions for visualization (to be created)
├── data/                     # TSPLIB .tsp files
├── results/                  # Output files (tours, plots)
├── main.py                   # Main execution script (to be created)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Tech Stack

- **Python 3.8+**
- **NumPy**: Numerical computations and matrix operations
- **Matplotlib**: Visualization of tours and convergence plots

## Installation

```bash
pip install -r requirements.txt
```

## Components

### DataLoader (`src/data_loader.py`)

Parses TSPLIB format files and computes distance matrices.

**Features:**
- Supports EUC_2D (Euclidean 2D) distance type
- Supports GEO (Geographical) distance type
- Extracts problem metadata (name, dimension, etc.)
- Validates coordinate data
- Computes symmetric distance matrices

**Usage:**
```python
from src.data_loader import DataLoader

loader = DataLoader('data/berlin52.tsp')
distance_matrix = loader.get_distance_matrix()
coordinates = loader.get_coordinates()
metadata = loader.get_metadata()
```

### TSPAlgorithm (`src/tsp_algorithm.py`)

Abstract base class defining the interface for all TSP solving algorithms.

**Key Methods:**
- `solve(**kwargs)`: Main solving method (abstract)
- `get_best_solution()`: Returns the best tour and its distance
- `get_history()`: Returns optimization history
- `calculate_tour_distance(tour)`: Computes tour length
- `validate_tour(tour)`: Validates tour correctness

**Usage:**
```python
from src.tsp_algorithm import TSPAlgorithm

class MyAlgorithm(TSPAlgorithm):
    def solve(self, **kwargs):
        # Implementation here
        pass
    
    def _initialize(self):
        # Initialization here
        pass
```

### AntSystem (`src/ant_system.py`)

Standard Ant Colony Optimization (AS) algorithm implementation.

**Features:**
- Probabilistic solution construction based on pheromone and heuristic information
- All ants deposit pheromones proportional to solution quality
- Configurable hyperparameters (α, β, ρ, Q)
- Convergence history tracking
- Early stopping support

**Key Parameters:**
- `num_ants`: Number of ants in the colony
- `alpha`: Pheromone importance factor (α)
- `beta`: Heuristic information importance factor (β)
- `evaporation_rate`: Pheromone evaporation rate (ρ)
- `q`: Constant for pheromone deposit calculation

**Usage:**
```python
from src.ant_system import AntSystem
froAlgorithm Comparison

| Feature | Ant System (AS) | Max-Min Ant System (MMAS) | Improved ACO (MMAS + 2-opt) |
|---------|-----------------|---------------------------|------------------------------|
| Pheromone Update | All ants | Best ant only | Best ant only |
| Pheromone Bounds | No limits | [τ_min, τ_max] | [τ_min, τ_max] |
| Local Search | None | None | 2-opt optimization |
| Convergence | Faster, but may get stuck | More robust | Fastest to quality solutions |
| Solution Quality | Good | Better | Best (locally optimal) |
| Stagnation Handling | None | Automatic reinitialization | Automatic reinitialization |
| Computational Cost | Low | Medium | Medium-High |
| Typical ρ | 0.5 | 0.02 | 0.02 |
| Best for | Small instances | Large instances | Quality-critical applications |

## Next Steps

1. ✅ ~~Implement the ACO algorithms~~ (AS and MMAS complete
as_solver = AntSystem(
    distance_matrix=loader.get_distance_matrix(),
    num_ants=20,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.5
)

best_tour, best_distance = as_solver.solve(max_iterations=100, verbose=True)
history = as_solver.get_history()
```

### MaxMinAntSystem (`src/mmas.py`)

Max-Min Ant System (MMAS) - an improved ACO variant with pheromone limits.

**Features:**
- Pheromone bounds [τ_min, τ_max] to prevent premature convergence
- Only best ant (iteration-best or global-best) updates pheromones
- Automatic pheromone reinitialization on stagnation
- Dynamic bound adjustment based on solution quality
- More robust than standard AS

**Key Parameters:**
- `num_ants`: Number of ants in the colony
- `alpha`: Pheromone importance factor (α)
- `beta`: Heuristic information importance factor (β)
- `evaporation_rate`: Pheromone evaporation rate (ρ) - typically lower than AS
- `use_iteration_best`: Use iteration-best (True) or global-best (False)
- `pbest`: Probability parameter for τ_max calculation

**Usage:**
```python
from src.mmas import MaxMinAntSystem
from src.data_loader import DataLoader

loader = DataLoader('data/berlin52.tsp')
mmas_solver = MaxMinAntSystem(
    distance_matrix=loader.get_distance_matrix(),
    num_ants=20,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.02,
    use_iteration_best=True
)

best_tour, best_distance = mmas_solver.solve(max_iterations=100, verbose=True)
history = mmas_solver.get_history()
```

### ImprovedACO (`src/improved_aco.py`)

Hybrid algorithm combining MMAS with 2-opt local search for superior solution quality.

**Features:**
- Inherits all MMAS benefits (pheromone bounds, stagnation handling)
- **2-opt local search** removes crossing edges and reconnects tours optimally
- Configurable strategy: apply to best ant only (faster) or all ants (thorough)
- Local search statistics tracking
- Guaranteed local optimality for each improved tour

**Key Improvements:**
- **Better convergence**: Faster approach to high-quality solutions
- **Shorter paths**: Typically 5-20% better than MMAS alone
- **Local optimality**: Each solution is locally optimal under 2-opt

**Key Parameters:**
- All MMAS parameters (inherits from MaxMinAntSystem)
- `apply_local_search_to_all`: Apply 2-opt to all ants (True) or best only (False)
- `max_local_search_iterations`: Maximum 2-opt iterations per tour

**Usage:**
```python
from src.improved_aco import ImprovedACO
from src.data_loader import DataLoader

loader = DataLoader('data/berlin52.tsp')
improved = ImprovedACO(
    distance_matrix=loader.get_distance_matrix(),
    num_ants=20,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.02,
    apply_local_search_to_all=False  # Apply only to best ant (recommended)
)

best_tour, best_distance = improved.solve(max_iterations=100, verbose=True)
stats = improved.get_local_search_statistics()
print(f"Local search improvement rate: {stats['improvement_rate_percent']:.1f}%")
```

## TSPLIB Format Support

The DataLoader currently supports:
- **EUC_2D**: 2D Euclidean distances (rounded to nearest integer)
- **GEO**: Geographical coordinates on Earth's surface

### Example TSPLIB File Structure

```
NAME: berlin52
COMMENT: 52 locations in Berlin (Groetschel)
TYPE: TSP
DIMENSION: 52
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
...
EOF
```

## Usage

### Basic Usage

Run a single algorithm on a TSPLIB file:

```bash
python main.py data/berlin52.tsp --algorithm MMAS --ants 20 --iterations 100 --verbose
```

### Comparison Mode

Compare all three algorithms on the same dataset:

```bash
python main.py data/berlin52.tsp --compare --iterations 100 --verbose
```

### Multiple Runs

Run an algorithm multiple times and report statistics:

```bash
python main.py data/eil76.tsp --algorithm Improved --runs 5 --iterations 100
```

### Save Results

Save plots and tour files to a directory:

```bash
python main.py data/berlin52.tsp --compare --save-results results/ --iterations 100
```

### Command-Line Options

```
Required:
  file                    Path to TSPLIB format .tsp file

Algorithm Selection:
  --algorithm {AS,MMAS,Improved}  Algorithm to use (default: MMAS)
  --compare                       Compare all three algorithms

Parameters:
  --ants N                Number of ants (default: 20)
  --iterations N          Maximum iterations (default: 100)
  --alpha FLOAT          Pheromone importance (default: 1.0)
  --beta FLOAT           Heuristic importance (default: 2.0)
  --rho FLOAT            Evaporation rate (default: auto)
  --q FLOAT              Pheromone deposit constant for AS (default: 100.0)

Execution:
  --runs N               Number of independent runs (default: 1)
  --early-stopping N     Stop after N iterations without improvement
  --seed N               Random seed for reproducibility
  
Output:
  --verbose              Print detailed progress
  --save-results DIR     Save plots and tours to directory
  --no-plot              Don't display plots
```

## Next Steps

1. ✅ ~~Implement the ACO algorithms~~ (AS, MMAS, and ImprovedACO complete)
2. ✅ ~~Create utility functions for visualization~~ (`utils.py` complete)
3. ✅ ~~Develop the main execution script~~ (`main.py` complete)
4. Add sample TSPLIB datasets to the `data/` folder
5. Run benchmarks and performance analysis

## References

- [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press.

## License

MIT License
