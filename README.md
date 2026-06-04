# Multidimensional Knapsack Problem Solver using Genetic Algorithm

This project solves the **Multidimensional Knapsack Problem** using a Genetic Algorithm with **PyGAD**. It includes direct integration with **OR-Library** test datasets.

## Problem Description

The multidimensional knapsack problem involves:
- **Selecting items** to maximize total value
- **Respecting multiple constraints** (weight, volume, cost, etc.)
- Each item has one value and multiple weights (one per dimension)

## Features

- ✅ Supports multiple weight dimensions (2D, 3D, nD knapsack)
- ✅ Genetic Algorithm-based optimization using PyGAD
- ✅ Flexible fitness function with constraint violation penalties
- ✅ **Direct integration with OR-Library test datasets**
- ✅ Parse and solve standard benchmark problems
- ✅ Detailed solution reporting and analysis
- ✅ Easily customizable for different problem instances

## Installation

1. Install Python 3.7+
2. Install required packages:
```bash
pip install -r requirements.txt
```

Note: If you don't have pandas installed yet, add it:
```bash
pip install pandas
```

## Usage

### Option 1: Run with OR-Library Data (Recommended)

The solver now automatically downloads test data from the OR-Library and solves real benchmark problems.

```bash
# Solve the first problem from mknapcb1 dataset
python knapsack.py
```

This will:
1. Download the mknapcb1.txt file from OR-Library
2. Parse the test problems
3. Solve the first problem using genetic algorithm
4. Show results and compare with optimal value if available

### Option 2: Solve Multiple Problems

Use the advanced solver to solve multiple test instances:

```bash
python solve_or_library.py
```

This will:
- Download OR-Library test data
- Solve multiple problems
- Create a summary table with results
- Show optimality gaps

### Option 3: Use Custom Data

```python
from knapsack import KnapsackProblem, solve_knapsack

# Define custom problem
values = [10, 20, 15, 25, 30]
weights = [[2, 3], [5, 4], [3, 5], [4, 6], [6, 5]]
capacities = [20, 25]

# Create and solve
problem = KnapsackProblem(values, weights, capacities)
ga, solution, items, value = solve_knapsack(problem)
```

## OR-Library Datasets

This solver integrates with the **OR-Library** which provides standard benchmark test instances:

### Available Files

- **mknap1.txt** - 7 test problems (small, from Petersen 1967)
- **mknap2.txt** - 48 test problems (medium)
- **mknapcb1.txt to mknapcb9.txt** - 30 problems each (100 items, 5 constraints)

### mknapcb Files Structure

Each file contains 30 problems with varying tightness ratios:
- **Problems 1-10**: Tightness ratio 0.25 (loose constraints)
- **Problems 11-20**: Tightness ratio 0.50 (medium constraints)
- **Problems 21-30**: Tightness ratio 0.75 (tight constraints)

Higher tightness = more challenging problem

### Reference

OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html

## Algorithm Parameters

- **`num_generations`**: Number of iterations (default: 100)
- **`population_size`**: Size of population per generation (default: 50)
- **`seed`**: Random seed for reproducibility (default: 42)

For harder problems (tight constraints), consider:
- Increasing `num_generations` to 200-500
- Increasing `population_size` to 100-200
- Adjusting penalty multiplier in fitness function

## Fitness Function

The fitness function:
1. Calculates total value from selected items
2. Checks weight constraints for each dimension
3. Applies penalties for exceeding capacity
4. Returns fitness = value - penalty

## Output Example

```
Problem 1 | Items: 100 | Constraints: 5 | Optimal: 13727

Generation 0 | Best Fitness = 11500.00
Generation 1 | Best Fitness = 12200.50
...
Generation 99 | Best Fitness = 13500.75

======================================================================
SOLUTION DETAILS
======================================================================
Selected items (indices): [0, 3, 5, 7, 12, ...]
Number of items selected: 45
Total value: 13500

Weight Usage by Dimension:
  Dimension 0: 19.95/20 (99.8%) ✓ OK
  Dimension 1: 24.99/25 (99.96%) ✓ OK
  Dimension 2: 19.90/20 (99.5%) ✓ OK
  Dimension 3: 24.80/25 (99.2%) ✓ OK
  Dimension 4: 19.99/20 (99.95%) ✓ OK

Feasible solution: Yes ✓
Optimality Gap: 1.64%
======================================================================
```

## Advanced Usage

### Parse Custom OR-Library Format

```python
from knapsack import parse_or_library_format

# Parse raw data in OR-Library format
problems = parse_or_library_format(data_text)

for problem in problems:
    print(f"Problem {problem['problem_id']}: {problem['num_items']} items")
```

### Download Different Datasets

```python
from knapsack import download_or_library_data

# Download different mknapcb files
problems = download_or_library_data("mknapcb2")  # File 2
problems = download_or_library_data("mknap1")    # Traditional format
```

### Customize Solver Parameters

```python
# Solve with custom parameters
ga, solution, items, value = solve_knapsack(
    problem,
    num_generations=300,      # More iterations
    population_size=100,      # Larger population
    seed=123
)
```

## Theory

- **Binary Encoding**: Each item represented by a binary gene (0 or 1)
- **Selection**: Tournament selection with K=3
- **Crossover**: Uniform crossover operator
- **Mutation**: Random mutation on 10% of genes
- **Constraint Handling**: Penalty-based approach for infeasible solutions

## Performance Tips

1. **For easy problems** (low tightness, < 50 items):
   - 50-100 generations
   - 30-50 population size
   
2. **For medium problems** (50-200 items):
   - 100-200 generations
   - 50-100 population size
   
3. **For hard problems** (tight constraints, > 200 items):
   - 300-500 generations
   - 100-200+ population size
   - Consider increasing penalty multiplier

## Troubleshooting

### No Feasible Solution Found

- Increase penalty multiplier in fitness function
- Increase population size and generations
- Check if constraints are too tight

### Network Error Downloading Data

- Check internet connection
- Verify URL is correct
- Download file manually and use custom data

### Slow Performance

- Reduce `num_generations` for faster results
- Reduce `population_size`
- Use smaller problem instance

## References

- PyGAD Documentation: https://pygad.readthedocs.io/
- OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/
- Genetic Algorithms: https://en.wikipedia.org/wiki/Genetic_algorithm
- Knapsack Problem: https://en.wikipedia.org/wiki/Knapsack_problem
- Original Paper: Chu, P.C. and Beasley, J.E. "A genetic algorithm for the multidimensional knapsack problem", Journal of Heuristics, vol. 4, 1998, pp63-86

## Author

Created for solving multidimensional knapsack optimization problems using evolutionary algorithms with OR-Library benchmark datasets.
