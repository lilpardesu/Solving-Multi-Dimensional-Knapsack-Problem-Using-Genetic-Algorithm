# Multidimensional Knapsack Problem Solver

A genetic algorithm solver for the **Multidimensional Knapsack Problem (MKP)** using [PyGAD](https://pygad.readthedocs.io/). It fetches benchmark instances from the OR-Library, evolves solutions across multiple adaptive runs, and stops automatically when results are good enough.

---

## What is the Multidimensional Knapsack Problem?

The classic knapsack problem asks: given a set of items, each with a value and a weight, which items should you pack to maximize total value without exceeding a weight limit?

The **multidimensional** variant adds multiple simultaneous constraints — for example, weight, volume, and cost all at once. Each item must satisfy every constraint at the same time, making the problem significantly harder.

---

## How it works

The solver runs a genetic algorithm (GA) that evolves a population of binary solutions (1 = item selected, 0 = not selected). Each individual is scored by a **fitness function** that rewards high total value and penalizes constraint violations.

Rather than running a fixed number of generations, it uses an **adaptive loop**:

1. Run the GA for up to N generations
2. After each run, check whether the result is good enough:
   - Is the solution within 2% of the known optimal value?
   - Has improvement stagnated for the last 30 generations?
   - Has the total generation budget been exhausted?
3. If none of those conditions are met, start another run with a different random seed
4. Repeat until a stopping condition is triggered

This means the solver automatically does more work on hard problems and stops early on easy ones.

---

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:

```
pygad==3.0.0
numpy>=1.21.0
pandas>=1.3.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the solver directly:

```bash
python solution.py
```

On startup it will:

1. Download `mknapcb1.txt` from the OR-Library (10 benchmark MKP instances)
2. Solve the first problem using the adaptive GA
3. Print per-generation progress, convergence reports, and a final solution summary

### Configurable parameters

These can be adjusted in the `solve_knapsack_adaptive(...)` call at the bottom of `solution.py`:

| Parameter | Default | Description |
|---|---|---|
| `generations_per_run` | `100` | Max generations per GA round |
| `population_size` | `50` | Number of solutions per generation |
| `seed` | `42` | Initial random seed (increments each run) |
| `gap_threshold` | `0.02` | Stop when within this fraction of optimal (2%) |
| `stagnation_window` | `30` | Generations without improvement before stopping |
| `max_total_generations` | `1000` | Hard cap across all runs |

---

## Project structure

```
.
├── solution.py        # Main solver (all logic in one file)
└── requirements.txt   # Python dependencies
```

### Key components inside `solution.py`

| Component | Role |
|---|---|
| `download_or_library_data` | Fetches benchmark `.txt` files from OR-Library by name |
| `parse_or_library_format` | Parses the integer-token format used by OR-Library |
| `KnapsackProblem` | Wraps problem data; provides the fitness function and solution decoder |
| `ConvergenceTracker` | Tracks fitness history; decides whether to stop or run more generations |
| `callback_generation` | PyGAD hook called every generation; feeds the tracker and triggers early stops |
| `solve_knapsack_adaptive` | Outer loop that launches GA runs until the tracker is satisfied |

---

## Output

A typical run prints:

```
>>> Starting run #1 (up to 100 generations, seed: 42)
  Gen    1 | Best Fitness = 3214.00
  Gen   10 | Best Fitness = 3501.00
  ...
  ────────────────────────────────────────────────────────────
  Convergence Report (after run #1)
  ────────────────────────────────────────────────────────────
  Total generations run : 100
  Best fitness so far   : 3720.00
  Optimality gap        : 1.85%
  Stagnated             : No
  ────────────────────────────────────────────────────────────
```

Followed by a final solution summary showing selected items, weight usage per dimension, and feasibility status.

---

## Data source

Test instances come from the [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html) maintained by J.E. Beasley. The default file `mknapcb1.txt` contains 10 problems with 100 items and 5 constraints each, with known optimal values included for benchmarking.

---

## Algorithm details

| GA setting | Value |
|---|---|
| Selection | Tournament (k=3) |
| Crossover | Uniform |
| Mutation | Random, ~10% of genes per individual |
| Gene type | Integer (0 or 1) |
| Parent pool | 50% of population |

Infeasible solutions are not discarded — they receive a penalty proportional to how much they exceed each capacity, allowing the GA to evolve toward feasibility gradually.