import numpy as np
import pygad
import urllib.request

# ============================================================================
# Multidimensional Knapsack Problem Solver using Genetic Algorithm (PyGAD)
# ============================================================================

def download_or_library_data(problem_file="mknapcb1"):
    """
    Download OR-Library test data for multidimensional knapsack problems.
    """
    url = f"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{problem_file}.txt"
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        return parse_or_library_format(data)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def parse_or_library_format(data_text):
    # Parse all integers regardless of line breaks
    tokens = [int(x) for x in data_text.split()]
    idx = 0

    num_problems = tokens[idx]
    idx += 1
    problems = []

    for p in range(num_problems):
        n = tokens[idx]
        m = tokens[idx + 1]
        optimal = tokens[idx + 2]
        idx += 3

        values = tokens[idx:idx + n]
        idx += n

        weights = []
        for _ in range(m):
            row = tokens[idx:idx + n]
            idx += n
            weights.append(row)

        capacities = tokens[idx:idx + m]
        idx += m

        problems.append({
            "num_items": n,
            "num_constraints": m,
            "optimal_value": optimal,
            "values": values,
            "weights": weights,
            "capacities": capacities,
            "problem_id": p + 1
        })

    return problems



# ============================================================================
# Convergence Tracker
# ============================================================================

class ConvergenceTracker:
    """
    Tracks GA progress across runs and decides whether more generations
    are needed based on improvement rate and optimality gap.
    """

    def __init__(
        self,
        optimal_value=None,
        gap_threshold=0.02,
        stagnation_window=30,
        min_improvement=0.001,
        max_total_generations=1000,
    ):
        self.optimal_value = optimal_value
        self.gap_threshold = gap_threshold
        self.stagnation_window = stagnation_window
        self.min_improvement = min_improvement
        self.max_total_generations = max_total_generations

        self.all_fitness_history = []
        self.total_generations_run = 0
        self.run_count = 0
        self.best_fitness_ever = -np.inf

        # Live tracking within current run
        self._live_history = []

    def update_live(self, current_best_fitness):
        """Called every generation from the callback."""
        self._live_history.append(current_best_fitness)
        self.all_fitness_history.append(current_best_fitness)
        self.total_generations_run += 1

        if current_best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = current_best_fitness

    def should_stop_early(self):
        """
        Called every generation — returns True if GA should stop NOW.
        Checks hard cap, optimality gap, and stagnation.
        """
        # 1. Hard cap on total generations
        if self.total_generations_run >= self.max_total_generations:
            return True

        # 2. Optimality gap — stop if close enough
        gap = self.optimality_gap()
        if gap is not None and gap <= self.gap_threshold:
            return True

        # 3. Stagnation within combined history
        if self.is_stagnated():
            return True

        return False

    def finalize_run(self):
        """Call after each GA run completes (or is stopped)."""
        self.run_count += 1
        self._live_history = []  # Reset live buffer for next run

    def optimality_gap(self):
        """Return gap vs known optimal (None if unknown)."""
        if self.optimal_value and self.optimal_value > 0:
            return (self.optimal_value - self.best_fitness_ever) / self.optimal_value
        return None

    def is_stagnated(self):
        """True if fitness hasn't improved meaningfully in the last window."""
        history = self.all_fitness_history
        if len(history) < self.stagnation_window:
            return False
        recent = history[-self.stagnation_window:]
        window_best = max(recent)
        window_start = recent[0]
        if window_start <= 0:
            return False
        improvement = (window_best - window_start) / abs(window_start)
        return improvement < self.min_improvement

    def should_continue(self):
        """Between-run decision — returns (continue: bool, reason: str)."""
        if self.total_generations_run >= self.max_total_generations:
            return False, f"Reached max total generations ({self.max_total_generations})"

        gap = self.optimality_gap()
        if gap is not None:
            if gap <= 0:
                return False, "Optimal solution found!"
            if gap <= self.gap_threshold:
                return False, f"Within {self.gap_threshold*100:.1f}% of optimal (gap={gap*100:.2f}%)"

        if self.is_stagnated():
            return False, f"Stagnated: no improvement in last {self.stagnation_window} generations"

        return True, "Still improving — running more generations"

    def report(self):
        """Print a summary of progress so far."""
        gap = self.optimality_gap()
        print(f"\n{'─'*60}")
        print(f"  Convergence Report (after run #{self.run_count})")
        print(f"{'─'*60}")
        print(f"  Total generations run : {self.total_generations_run}")
        print(f"  Best fitness so far   : {self.best_fitness_ever:.2f}")
        if gap is not None:
            print(f"  Optimality gap        : {gap*100:.2f}%")
        else:
            print(f"  Optimality gap        : unknown (no reference)")
        print(f"  Stagnated             : {'Yes' if self.is_stagnated() else 'No'}")
        print(f"{'─'*60}\n")


# ============================================================================
# Knapsack Problem
# ============================================================================

class KnapsackProblem:
    """
    Multidimensional Knapsack Problem:
    - Items have a value and multiple weights (dimensions)
    - Maximize total value while respecting all weight constraints
    """

    def __init__(self, values, weights, capacities, problem_id=None, optimal_value=None):
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)
        self.num_items = len(values)
        self.num_dimensions = len(capacities)
        self.problem_id = problem_id
        self.optimal_value = optimal_value

        print(f"Problem initialized:")
        if problem_id:
            print(f"  Problem ID      : {problem_id}")
        print(f"  Items           : {self.num_items}")
        print(f"  Dimensions      : {self.num_dimensions}")
        if optimal_value and optimal_value > 0:
            print(f"  Known optimal   : {optimal_value}")
        print()

    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        Fitness function for the genetic algorithm.
        Maximizes value while penalizing constraint violations.
        """
        total_value = np.sum(solution * self.values)
        total_weights = np.dot(self.weights, solution)

        penalty = 0
        for dim in range(self.num_dimensions):
            if total_weights[dim] > self.capacities[dim]:
                excess = total_weights[dim] - self.capacities[dim]
                penalty += excess * 10

        return total_value - penalty

    def decode_solution(self, solution):
        """Decode solution to show selected items and statistics."""
        selected_items = solution.astype(int)
        selected_indices = np.where(selected_items == 1)[0]
        total_value = np.sum(selected_items * self.values)
        total_weights = np.dot(self.weights, selected_items)

        print("\n" + "="*70)
        print("SOLUTION DETAILS")
        print("="*70)
        print(f"Selected items   : {selected_indices.tolist()}")
        print(f"Items count      : {len(selected_indices)}")
        print(f"Total value      : {total_value}")
        print("\nWeight Usage by Dimension:")
        for dim in range(self.num_dimensions):
            capacity = self.capacities[dim]
            used = total_weights[dim]
            pct = (used / capacity) * 100
            status = "✓" if used <= capacity else "✗ VIOLATED"
            print(f"  Dim {dim:>2}: {used:>8.1f} / {capacity} ({pct:5.1f}%) {status}")

        is_feasible = np.all(total_weights <= self.capacities)
        print(f"\nFeasible: {'Yes ✓' if is_feasible else 'No ✗'}")
        print("="*70)

        return selected_indices, total_value, total_weights


# ============================================================================
# Callback
# ============================================================================

def callback_generation(ga_instance):
    """Callback called every generation — checks early stopping via tracker."""
    gen = ga_instance.generations_completed
    best = ga_instance.best_solutions_fitness[-1]

    if gen % 10 == 0 or gen == 1:
        print(f"  Gen {gen:>4} | Best Fitness = {best:.2f}")

    # Update tracker with this generation's best fitness
    ga_instance.tracker.update_live(best)

    # Ask tracker if we should stop early
    if ga_instance.tracker.should_stop_early():
        print(f"  [Early stop triggered at gen {gen}]")
        return "stop"


# ============================================================================
# Adaptive Solver
# ============================================================================

def solve_knapsack_adaptive(
    problem,
    generations_per_run=100,
    population_size=50,
    seed=42,
    gap_threshold=0.02,
    stagnation_window=30,
    max_total_generations=1000,
):
    """
    Runs the GA in multiple rounds. After each round, ConvergenceTracker
    decides whether results are good enough or more generations are needed.

    Parameters:
    - problem               : KnapsackProblem instance
    - generations_per_run   : Max generations per GA round
    - population_size       : Population size
    - seed                  : Random seed
    - gap_threshold         : Stop when within this fraction of optimal (e.g. 0.02 = 2%)
    - stagnation_window     : Generations to look back for stagnation check
    - max_total_generations : Hard cap across all runs combined
    """

    tracker = ConvergenceTracker(
        optimal_value=problem.optimal_value,
        gap_threshold=gap_threshold,
        stagnation_window=stagnation_window,
        max_total_generations=max_total_generations,
    )

    best_solution_ever = None
    best_fitness_ever = -np.inf
    current_seed = seed

    print("="*70)
    print("ADAPTIVE GENETIC ALGORITHM SOLVER")
    print("="*70)

    while True:
        # Check before starting a new run
        keep_going, reason = tracker.should_continue()
        if not keep_going:
            print(f"\n■ Stopping before new run: {reason}")
            break

        print(f"\n>>> Starting run #{tracker.run_count + 1} "
              f"(up to {generations_per_run} generations, seed: {current_seed})")

        ga = pygad.GA(
            num_generations=generations_per_run,
            num_parents_mating=max(2, int(population_size * 0.5)),
            fitness_func=problem.fitness_function,
            sol_per_pop=population_size,
            num_genes=problem.num_items,
            gene_type=int,
            init_range_low=0,
            init_range_high=2,
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="uniform",
            mutation_type="random",
            mutation_num_genes=max(1, problem.num_items // 10),
            allow_duplicate_genes=True,
            random_seed=current_seed,
            on_generation=callback_generation,
        )

        # Attach tracker so the callback can access it
        ga.tracker = tracker

        ga.run()

        tracker.finalize_run()

        # Check if this run produced a better solution
        solution, fitness, _ = ga.best_solution()
        if fitness > best_fitness_ever:
            best_fitness_ever = fitness
            best_solution_ever = solution.copy()

        tracker.report()

        keep_going, reason = tracker.should_continue()
        print(f"  Decision: {'▶ Continue' if keep_going else '■ Stop'}")
        print(f"  Reason  : {reason}\n")

        if not keep_going:
            break

        # Vary seed so each run explores differently
        current_seed += 1

    print("="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Total runs        : {tracker.run_count}")
    print(f"Total generations : {tracker.total_generations_run}")
    print(f"Best fitness      : {best_fitness_ever:.2f}")

    selected_indices, total_value, total_weights = problem.decode_solution(best_solution_ever)

    if problem.optimal_value and problem.optimal_value > 0:
        gap = (problem.optimal_value - total_value) / problem.optimal_value * 100
        print(f"\nOptimality gap    : {gap:.2f}%")

    return best_solution_ever, selected_indices, total_value, tracker


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Downloading OR-Library test data (mknapcb1.txt)...")
    problems = download_or_library_data("mknapcb1")

    if not problems:
        print("ERROR: Failed to download data.")
        exit(1)

    problem_data = problems[0]

    print("\n" + "="*70)
    print("MULTIDIMENSIONAL KNAPSACK — OR-LIBRARY TEST DATA")
    print("="*70)
    print(f"  Problem ID      : {problem_data['problem_id']}")
    print(f"  Items           : {problem_data['num_items']}")
    print(f"  Constraints     : {problem_data['num_constraints']}")
    if problem_data['optimal_value'] > 0:
        print(f"  Known Optimal   : {problem_data['optimal_value']}")
    print(f"  Capacities      : {problem_data['capacities']}")
    print("="*70)

    problem = KnapsackProblem(
        values=problem_data['values'],
        weights=problem_data['weights'],
        capacities=problem_data['capacities'],
        problem_id=problem_data['problem_id'],
        optimal_value=problem_data['optimal_value'],
    )

    best_solution, selected_items, total_value, tracker = solve_knapsack_adaptive(
        problem,
        generations_per_run=100,    # Max generations per round
        population_size=50,
        seed=42,
        gap_threshold=0.02,         # Stop if within 2% of optimal
        stagnation_window=30,       # Stop if no improvement for 30 gens
        max_total_generations=1000  # Never exceed 1000 total generations
    )
