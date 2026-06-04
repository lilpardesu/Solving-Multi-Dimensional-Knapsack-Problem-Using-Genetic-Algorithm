import numpy as np
import pygad
import random
import urllib.request
import io

# ============================================================================
# Multidimensional Knapsack Problem Solver using Genetic Algorithm (PyGAD)
# ============================================================================

def download_or_library_data(problem_file="mknapcb1"):
    """
    Download OR-Library test data for multidimensional knapsack problems.
    
    Parameters:
    - problem_file: name of the file (e.g., "mknapcb1", "mknapcb2", etc.)
    
    Returns:
    - List of test problems with their data
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
    """
    Parse OR-Library multidimensional knapsack problem format.
    
    Format:
    - First line: number of test problems (K)
    - For each problem:
      - n (variables), m (constraints), optimal value
      - p(j) for j=1,...,n (item values)
      - For each constraint i: r(i,j) for j=1,...,n (item weights)
      - b(i) for i=1,...,m (capacity limits)
    """
    lines = data_text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    problems = []
    idx = 0
    
    # First line is number of problems
    num_problems = int(lines[idx])
    idx += 1
    
    for p in range(num_problems):
        # Parse problem header
        parts = lines[idx].split()
        n = int(parts[0])  # number of items
        m = int(parts[1])  # number of constraints
        optimal = int(parts[2]) if len(parts) > 2 else 0
        idx += 1
        
        # Parse item values
        values = []
        while len(values) < n:
            values.extend([int(x) for x in lines[idx].split()])
            idx += 1
        values = values[:n]
        
        # Parse weights (constraints) and capacities
        weights = []
        for i in range(m):
            constraint = []
            while len(constraint) < n:
                constraint.extend([int(x) for x in lines[idx].split()])
                idx += 1
            weights.append(constraint[:n])
        
        # Parse capacities
        capacities = []
        while len(capacities) < m:
            capacities.extend([int(x) for x in lines[idx].split()])
            idx += 1
        capacities = capacities[:m]
        
        problem = {
            'num_items': n,
            'num_constraints': m,
            'optimal_value': optimal,
            'values': values,
            'weights': weights,
            'capacities': capacities,
            'problem_id': p + 1
        }
        problems.append(problem)
    
    return problems


# Problem definition
class KnapsackProblem:
    """
    Multidimensional Knapsack Problem:
    - Items have a value and multiple weights (dimensions)
    - Maximize total value while respecting all weight constraints
    """
    
    def __init__(self, values, weights, capacities, problem_id=None, optimal_value=None):
        """
        Parameters:
        - values: list of values for each item
        - weights: list of lists, where weights[i] = [weight_dim1, weight_dim2, ...]
        - capacities: list of capacity limits for each dimension
        - problem_id: optional problem identifier
        - optimal_value: optional known optimal value
        """
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)
        self.num_items = len(values)
        self.num_dimensions = len(capacities)
        self.problem_id = problem_id
        self.optimal_value = optimal_value
        
        print(f"Problem initialized:")
        if problem_id:
            print(f"  Problem ID: {problem_id}")
        print(f"  Number of items: {self.num_items}")
        print(f"  Number of dimensions: {self.num_dimensions}")
        if optimal_value and optimal_value > 0:
            print(f"  Known optimal value: {optimal_value}")
        print()
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        Fitness function for the genetic algorithm.
        Maximizes value while penalizing constraint violations.
        """
        # Decode solution (binary vector indicating selected items)
        selected_items = solution
        
        # Calculate total value
        total_value = np.sum(selected_items * self.values)
        
        # Calculate total weights for each dimension (constraint)
        # weights shape: (num_constraints, num_items), selected_items shape: (num_items,)
        # Result: (num_constraints,)
        total_weights = np.dot(self.weights, selected_items)
        
        # Penalty for exceeding capacity constraints
        penalty = 0
        constraint_violations = 0
        for dim in range(self.num_dimensions):
            if total_weights[dim] > self.capacities[dim]:
                excess = total_weights[dim] - self.capacities[dim]
                # Proportional penalty based on excess weight
                penalty += excess * 10  # Penalty multiplier
                constraint_violations += 1
        
        # Final fitness = value - penalty
        fitness = total_value - penalty
        
        return fitness
    
    def decode_solution(self, solution):
        """Decode solution to show selected items and statistics."""
        selected_items = solution.astype(int)
        selected_indices = np.where(selected_items == 1)[0]
        
        total_value = np.sum(selected_items * self.values)
        total_weights = np.dot(self.weights, selected_items)
        
        print("\n" + "="*70)
        print("SOLUTION DETAILS")
        print("="*70)
        print(f"Selected items (indices): {selected_indices.tolist()}")
        print(f"Number of items selected: {len(selected_indices)}")
        print(f"Total value: {total_value}")
        print("\nWeight Usage by Dimension:")
        for dim in range(self.num_dimensions):
            capacity = self.capacities[dim]
            used = total_weights[dim]
            percentage = (used / capacity) * 100
            status = "✓ OK" if used <= capacity else "✗ VIOLATED"
            print(f"  Dimension {dim}: {used:.2f}/{capacity} ({percentage:.1f}%) {status}")
        
        is_feasible = np.all(total_weights <= self.capacities)
        print(f"\nFeasible solution: {'Yes ✓' if is_feasible else 'No ✗'}")
        print("="*70)
        
        return selected_indices, total_value, total_weights


def solve_knapsack(problem, num_generations=100, population_size=50, seed=42):
    """
    Solve the multidimensional knapsack problem using PyGAD.
    
    Parameters:
    - problem: KnapsackProblem instance
    - num_generations: Number of generations for GA
    - population_size: Size of population
    - seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Create genetic algorithm instance
    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=int(population_size * 0.5),
        fitness_func=problem.fitness_function,
        sol_per_pop=population_size,
        num_genes=problem.num_items,  # One gene per item (binary)
        gene_type=int,
        init_range_low=0,
        init_range_high=2,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        mutation_type="random",
        mutation_num_genes=10,
        allow_duplicate_genes=True,
        random_seed=seed,
        on_generation=callback_generation
    )
    
    # Run genetic algorithm
    ga.run()
    
    # Get best solution
    solution, fitness, solution_idx = ga.best_solution()
    
    print(f"\n{'='*70}")
    print(f"GENETIC ALGORITHM RESULTS")
    print(f"{'='*70}")
    print(f"Best fitness: {fitness:.2f}")
    print(f"Generations completed: {ga.generations_completed}")
    
    # Decode and display best solution
    selected_indices, total_value, total_weights = problem.decode_solution(solution)
    
    return ga, solution, selected_indices, total_value


def callback_generation(ga_instance):
    """Callback function to print generation progress."""
    print(f"Generation {ga_instance.generations_completed} | "
          f"Best Fitness = {ga_instance.best_solutions_fitness[-1]:.2f}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Downloading OR-Library test data (mknapcb1.txt)...")
    problems = download_or_library_data("mknapcb1")
    
    if not problems:
        print("ERROR: Failed to download OR-Library data. Please check your internet connection.")
        exit(1)
    
    # Solve the first problem as an example
    problem_data = problems[0]
    
    print("\n" + "="*70)
    print("MULTIDIMENSIONAL KNAPSACK PROBLEM - OR-LIBRARY TEST DATA")
    print("="*70)
    print("\nProblem Definition:")
    print(f"  Problem ID: {problem_data['problem_id']}")
    print(f"  Items: {problem_data['num_items']}")
    print(f"  Constraints: {problem_data['num_constraints']}")
    if problem_data['optimal_value'] > 0:
        print(f"  Known Optimal Value: {problem_data['optimal_value']}")
    print(f"  Item Values (first 10): {problem_data['values'][:10]}")
    print(f"  Capacity Limits: {problem_data['capacities']}")
    print("="*70)
    
    # Create problem instance
    problem = KnapsackProblem(
        values=problem_data['values'],
        weights=problem_data['weights'],
        capacities=problem_data['capacities'],
        problem_id=problem_data['problem_id'],
        optimal_value=problem_data['optimal_value']
    )
    
    # Solve using genetic algorithm
    ga, best_solution, selected_items, total_value = solve_knapsack(
        problem,
        num_generations=100,
        population_size=50,
        seed=42
    )
    
    # Compare with optimal value if available
    if problem.optimal_value > 0:
        gap = ((problem.optimal_value - total_value) / problem.optimal_value) * 100
        print(f"\nOptimality Gap: {gap:.2f}%")
    
    # Print selected items details
    print("\nSelected Items Details:")
    print(f"  Item indices: {selected_items.tolist()}")
    print(f"  Total value: {total_value}")
