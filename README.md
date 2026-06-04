```markdown
# Multidimensional Knapsack Problem Solver using Genetic Algorithm

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyGAD](https://img.shields.io/badge/PyGAD-latest-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

This project solves the **Multidimensional Knapsack Problem (MKP)** using an **Adaptive Genetic Algorithm** powered by [PyGAD](https://pygad.readthedocs.io/). It integrates directly with **OR-Library** benchmark datasets and features intelligent convergence tracking with automatic early stopping.

---

## Problem Description

The multidimensional knapsack problem involves:
- **Selecting items** to maximize total value
- **Respecting multiple weight constraints** (one per dimension)
- Each item has **one value** and **multiple weights** (one per dimension)

Formally:

```
Maximize:   Σ vᵢ · xᵢ
Subject to: Σ wᵢⱼ · xᵢ ≤ cⱼ   for all j = 1..m
            xᵢ ∈ {0, 1}
```

Where `vᵢ` = item value, `wᵢⱼ` = weight of item `i` in dimension `j`, `cⱼ` = capacity of dimension `j`.

---

## Features

- ✅ Supports multiple weight dimensions (2D, 3D, nD knapsack)
- ✅ **Adaptive multi-run GA** — automatically decides when to stop or continue
- ✅ **ConvergenceTracker** — monitors optimality gap, stagnation, and hard caps
- ✅ **Early stopping** within each generation via callback
- ✅ Penalty-based fitness function for constraint violation handling
- ✅ Direct integration with OR-Library benchmark datasets
- ✅ Detailed per-dimension weight usage reporting
- ✅ Reproducible runs via configurable random seed

---

## Project Structure

```
.
├── knapsack.py       # Main solver (all components in one file)
├── requirements.txt  # Python dependencies
└── README.md
```

### Key Components

| Component | Description |
|---|---|
| `download_or_library_data()` | Downloads benchmark data from OR-Library |
| `parse_or_library_format()` | Parses OR-Library `.txt` format into problem dicts |
| `KnapsackProblem` | Stores problem data; provides fitness function and solution decoder |
| `ConvergenceTracker` | Tracks GA progress; decides early stop and between-run continuation |
| `callback_generation()` | PyGAD generation callback; logs progress and triggers early stop |
| `solve_knapsack_adaptive()` | Main adaptive solver loop; runs multiple GA rounds intelligently |

---

## Installation

**Requirements:** Python 3.8+

1. Clone the repository:
```bash
git clone https://github.com/your-username/knapsack-ga.git
cd knapsack-ga
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pygad
numpy
```

---

## Usage

### Run the Solver (Default)

```bash
python knapsack.py
```

This will:
1. Download `mknapcb1.txt` from OR-Library automatically
2. Parse and load the first benchmark problem (100 items, 5 constraints)
3. Run the adaptive GA with convergence tracking
4. Print detailed results and optimality gap

---

### Use the Adaptive Solver in Your Code

```python
from knapsack import download_or_library_data, KnapsackProblem, solve_knapsack_adaptive

# Load OR-Library data
problems = download_or_library_data("mknapcb1")
data = problems[0]

# Create problem instance
problem = KnapsackProblem(
    values=data["values"],
    weights=data["weights"],
    capacities=data["capacities"],
    problem_id=data["problem_id"],
    optimal_value=data["optimal_value"],
)

# Run adaptive solver
best_solution, selected_items, total_value, tracker = solve_knapsack_adaptive(
    problem,
    generations_per_run=100,     # Max generations per GA round
    population_size=50,          # Population size
    seed=42,                     # Random seed
    gap_threshold=0.02,          # Stop if within 2% of optimal
    stagnation_window=30,        # Stop if no improvement for 30 gens
    max_total_generations=1000,  # Hard cap across all runs
)
```

---

### Use Custom Problem Data

```python
from knapsack import KnapsackProblem, solve_knapsack_adaptive

values     = [10, 20, 15, 25, 30]
weights    = [[2, 3], [5, 4], [3, 5], [4, 6], [6, 5]]
capacities = [20, 25]

problem = KnapsackProblem(values, weights, capacities)

best_solution, selected_items, total_value, tracker = solve_knapsack_adaptive(
    problem,
    generations_per_run=100,
    population_size=50,
)
```

---

### Parse Different OR-Library Files

```python
from knapsack import download_or_library_data

problems = download_or_library_data("mknapcb4")

for p in problems:
    print(f"Problem {p['problem_id']}: {p['num_items']} items, "
          f"{p['num_constraints']} dims, optimal={p['optimal_value']}")
```

---

## Adaptive Solver Parameters

| Parameter | Default | Description |
|---|---|---|
| `generations_per_run` | `100` | Max generations per GA round |
| `population_size` | `50` | Number of individuals per generation |
| `seed` | `42` | Initial random seed (incremented each run) |
| `gap_threshold` | `0.02` | Stop when within this fraction of optimal (2%) |
| `stagnation_window` | `30` | Generations without improvement to trigger stagnation stop |
| `max_total_generations` | `1000` | Hard cap on total generations across all runs |

---

## Convergence Tracker

The `ConvergenceTracker` class controls **when the solver stops**, both **within a run** (via the generation callback) and **between runs**.

### Stopping Conditions (checked in order)

| Condition | Description |
|---|---|
| **Hard cap** | `total_generations_run >= max_total_generations` |
| **Optimality gap** | Best fitness is within `gap_threshold` of known optimal |
| **Stagnation** | No meaningful improvement (`< 0.1%`) in last `stagnation_window` generations |
| **Optimal found** | Gap ≤ 0 (best equals or exceeds known optimal) |

### Convergence Report (printed after each run)

```
────────────────────────────────────────────────────────────
  Convergence Report (after run #2)
────────────────────────────────────────────────────────────
  Total generations run : 187
  Best fitness so far   : 13521.00
  Optimality gap        : 1.50%
  Stagnated             : No
────────────────────────────────────────────────────────────
```

---

## GA Configuration

| Parameter | Value | Notes |
|---|---|---|
| Gene type | `int` (0 or 1) | Binary encoding |
| Parent selection | Tournament (`K=3`) | Competitive selection |
| Crossover | Uniform | Even gene mixing |
| Mutation | Random | ~10% of genes per individual |
| Duplicate genes | Allowed | Binary genes may repeat |

---

## Fitness Function

```
fitness = Σ(vᵢ · xᵢ)  −  Σⱼ max(0, Σᵢ wᵢⱼ · xᵢ − cⱼ) × 10
```

- **Reward:** total value of selected items
- **Penalty:** `excess_weight × 10` for each violated dimension
- Penalty coefficient (`10`) can be tuned for tighter constraint enforcement

---

## OR-Library Datasets

| File | Problems | Items | Constraints | Notes |
|---|---|---|---|---|
| `mknap1.txt` | 7 | varies | varies | Small classic instances |
| `mknap2.txt` | 48 | varies | varies | Medium instances |
| `mknapcb1.txt` – `mknapcb9.txt` | 30 each | 100 | 5 | Standard benchmarks |

### Tightness Ratios in mknapcb Files

| Problems | Tightness | Difficulty |
|---|---|---|
| 1–10 | 0.25 | Easy (loose constraints) |
| 11–20 | 0.50 | Medium |
| 21–30 | 0.75 | Hard (tight constraints) |

> **Reference:** https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html

---

## Output Example

```
Downloading OR-Library test data (mknapcb1.txt)...

======================================================================
MULTIDIMENSIONAL KNAPSACK — OR-LIBRARY TEST DATA
======================================================================
  Problem ID      : 1
  Items           : 100
  Constraints     : 5
  Known Optimal   : 24381
  Capacities      : [6120, 8648, ...]
======================================================================

ADAPTIVE GENETIC ALGORITHM SOLVER
======================================================================

>>> Starting run #1 (up to 100 generations, seed: 42)
  Gen    1 | Best Fitness = 19800.00
  Gen   10 | Best Fitness = 22400.00
  Gen   50 | Best Fitness = 23800.00
  Gen  100 | Best Fitness = 24010.50

────────────────────────────────────────────────────────────
  Convergence Report (after run #1)
────────────────────────────────────────────────────────────
  Total generations run : 100
  Best fitness so far   : 24010.50
  Optimality gap        : 1.52%
  Stagnated             : No
────────────────────────────────────────────────────────────

>>> Starting run #2 (up to 100 generations, seed: 43)
  Gen   30 | Best Fitness = 24200.00
  [Early stop triggered at gen 47]

────────────────────────────────────────────────────────────
  Convergence Report (after run #2)
────────────────────────────────────────────────────────────
  Total generations run : 147
  Best fitness so far   : 24200.00
  Optimality gap        : 0.74%
  Stagnated             : Yes
────────────────────────────────────────────────────────────

■ Stopping: Stagnated: no improvement in last 30 generations

======================================================================
FINAL RESULT
======================================================================
Total runs        : 2
Total generations : 147
Best fitness      : 24200.00

======================================================================
SOLUTION DETAILS
======================================================================
Selected items   : [2, 5, 9, 14, ...]
Items count      : 47
Total value      : 24200

Weight Usage by Dimension:
  Dim  0:   6100.0 / 6120 ( 99.7%) ✓
  Dim  1:   8600.0 / 8648 ( 99.4%) ✓
  Dim  2:   7200.0 / 7314 ( 98.4%) ✓
  Dim  3:   5900.0 / 5916 ( 99.7%) ✓
  Dim  4:   6100.0 / 6254 ( 97.5%) ✓

Feasible: Yes ✓
======================================================================

Optimality gap    : 0.74%
```

---

## Performance Guidelines

| Problem Size | Recommended Settings |
|---|---|
| Small (< 50 items) | `generations_per_run=50`, `population_size=30` |
| Medium (50–200 items) | `generations_per_run=100`, `population_size=50` |
| Large (> 200 items) | `generations_per_run=300`, `population_size=100–200` |
| Tight constraints | Increase penalty multiplier; larger population |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| No feasible solution | Increase penalty multiplier (default `10`) in `fitness_function` |
| Slow convergence | Increase `population_size` or `max_total_generations` |
| Network error on download | Check internet connection; download file manually |
| Solver stops too early | Increase `stagnation_window` or `max_total_generations` |
| Solver never stops | Reduce `max_total_generations` or tighten `gap_threshold` |

---

## Theory

- **Binary Encoding** — each gene represents one item: `1` = selected, `0` = not selected
- **Tournament Selection** — `K=3` candidates compete; fittest parent wins
- **Uniform Crossover** — each gene independently inherited from either parent
- **Random Mutation** — approximately 10% of genes flipped per individual
- **Penalty Method** — infeasible solutions penalized proportionally to constraint excess
- **Adaptive Restart** — each run uses a different seed to escape local optima

---

## References

- PyGAD Documentation: https://pygad.readthedocs.io/
- OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/
- Genetic Algorithms: https://en.wikipedia.org/wiki/Genetic_algorithm
- Knapsack Problem: https://en.wikipedia.org/wiki/Knapsack_problem
- Chu, P.C. and Beasley, J.E. *"A genetic algorithm for the multidimensional knapsack problem"*, Journal of Heuristics, vol. 4, 1998, pp. 63–86