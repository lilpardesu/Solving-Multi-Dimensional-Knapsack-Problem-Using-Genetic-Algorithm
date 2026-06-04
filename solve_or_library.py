"""
OR-Library Multidimensional Knapsack Problem Solver
Solves multiple test instances from the OR-Library and reports results.

Reference: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html
"""

import sys
from knapsack import download_or_library_data, KnapsackProblem, solve_knapsack
import pandas as pd

def solve_multiple_problems(problem_file="mknapcb1", num_to_solve=None, 
                           num_generations=100, population_size=50):
    """
    Solve multiple test problems from OR-Library.
    
    Parameters:
    - problem_file: name of the OR-Library file (mknapcb1-9, mknap1-2, etc.)
    - num_to_solve: number of problems to solve (None = all)
    - num_generations: GA generations
    - population_size: GA population size
    """
    
    print(f"\n{'='*80}")
    print(f"SOLVING OR-LIBRARY MULTIDIMENSIONAL KNAPSACK PROBLEMS: {problem_file}")
    print(f"{'='*80}\n")
    
    # Download data
    print(f"Downloading {problem_file}.txt from OR-Library...")
    problems = download_or_library_data(problem_file)
    
    if not problems:
        print("Failed to download data.")
        return None
    
    # Limit number of problems to solve
    if num_to_solve:
        problems = problems[:num_to_solve]
    
    print(f"Successfully loaded {len(problems)} test problems.\n")
    
    # Results tracking
    results = []
    
    # Solve each problem
    for problem_data in problems:
        prob_id = problem_data['problem_id']
        n_items = problem_data['num_items']
        n_constraints = problem_data['num_constraints']
        optimal = problem_data['optimal_value']
        
        print(f"\n{'─'*80}")
        print(f"Problem {prob_id:2d} | Items: {n_items:3d} | Constraints: {n_constraints} | Optimal: {optimal}")
        print(f"{'─'*80}")
        
        try:
            # Create problem instance
            problem = KnapsackProblem(
                values=problem_data['values'],
                weights=problem_data['weights'],
                capacities=problem_data['capacities'],
                problem_id=prob_id,
                optimal_value=optimal
            )
            
            # Solve
            ga, best_solution, selected_items, total_value = solve_knapsack(
                problem,
                num_generations=num_generations,
                population_size=population_size,
                seed=42
            )
            
            # Calculate gap
            if optimal > 0:
                gap = ((optimal - total_value) / optimal) * 100
                is_optimal = gap < 0.01
            else:
                gap = None
                is_optimal = False
            
            # Store result
            result = {
                'Problem': prob_id,
                'Items': n_items,
                'Constraints': n_constraints,
                'Optimal': optimal if optimal > 0 else 'Unknown',
                'Found': int(total_value),
                'Gap (%)': f"{gap:.2f}" if gap is not None else "N/A",
                'Optimal?': '✓' if is_optimal else '✗'
            }
            results.append(result)
            
            print(f"\nResult:")
            print(f"  Found Value: {int(total_value)}")
            if gap is not None:
                print(f"  Optimality Gap: {gap:.2f}%")
            print(f"  Items Selected: {len(selected_items)} / {n_items}")
            
        except Exception as e:
            print(f"ERROR solving problem {prob_id}: {e}")
            continue
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Statistics
    if results:
        print(f"\n\nStatistics:")
        print(f"  Total problems solved: {len(results)}")
        
        optimal_counts = sum(1 for r in results if r['Optimal?'] == '✓')
        print(f"  Problems reaching optimal: {optimal_counts}")
        
        print(f"\n  Average solution quality: {len(results)} problems processed")
    
    return results, problems


def solve_single_problem(problem_file="mknapcb1", problem_num=1,
                        num_generations=100, population_size=50):
    """
    Solve a single test problem from OR-Library.
    
    Parameters:
    - problem_file: name of the OR-Library file
    - problem_num: which problem to solve (1-indexed)
    - num_generations: GA generations
    - population_size: GA population size
    """
    
    print(f"\n{'='*80}")
    print(f"SOLVING SINGLE OR-LIBRARY PROBLEM")
    print(f"{'='*80}\n")
    
    # Download data
    print(f"Downloading {problem_file}.txt from OR-Library...")
    problems = download_or_library_data(problem_file)
    
    if not problems or problem_num > len(problems):
        print(f"Error: Problem {problem_num} not found.")
        return None
    
    problem_data = problems[problem_num - 1]
    
    print("\n" + "="*70)
    print(f"PROBLEM {problem_num} DETAILS")
    print("="*70)
    print(f"\nProblem Information:")
    print(f"  Number of items: {problem_data['num_items']}")
    print(f"  Number of constraints: {problem_data['num_constraints']}")
    print(f"  Optimal value (known): {problem_data['optimal_value'] if problem_data['optimal_value'] > 0 else 'Unknown'}")
    print(f"\nItem values (first 20): {problem_data['values'][:20]}")
    print(f"Capacity limits: {problem_data['capacities']}")
    print("="*70)
    
    # Create and solve problem
    problem = KnapsackProblem(
        values=problem_data['values'],
        weights=problem_data['weights'],
        capacities=problem_data['capacities'],
        problem_id=problem_data['problem_id'],
        optimal_value=problem_data['optimal_value']
    )
    
    ga, best_solution, selected_items, total_value = solve_knapsack(
        problem,
        num_generations=num_generations,
        population_size=population_size,
        seed=42
    )
    
    # Print results
    print(f"\nSolution Summary:")
    print(f"  Total value: {int(total_value)}")
    print(f"  Items selected: {len(selected_items)}")
    print(f"  Items: {selected_items.tolist()}")
    
    if problem.optimal_value > 0:
        gap = ((problem.optimal_value - total_value) / problem.optimal_value) * 100
        print(f"  Optimality gap: {gap:.2f}%")
    
    return problem, best_solution, selected_items, total_value


if __name__ == "__main__":
    # Example: Solve first 3 problems from mknapcb1
    # Each problem has 100 items and 5 constraints
    
    print("\n" + "="*80)
    print("OR-LIBRARY MULTIDIMENSIONAL KNAPSACK PROBLEM SOLVER")
    print("="*80)
    print("\nAvailable test files from OR-Library:")
    print("  mknap1  - 7 problems (small)")
    print("  mknap2  - 48 problems (medium)")
    print("  mknapcb1-9 - 30 problems each (100 items, 5 constraints)")
    print("\nTightness ratios (for mknapcb files):")
    print("  Problems 1-10: tightness 0.25")
    print("  Problems 11-20: tightness 0.50")
    print("  Problems 21-30: tightness 0.75")
    
    # Solve multiple problems
    results, all_problems = solve_multiple_problems(
        problem_file="mknapcb1",
        num_to_solve=3,  # Solve first 3 problems as demo
        num_generations=50,  # Reduced for faster demo
        population_size=30
    )
    
    # Uncomment below to solve a single specific problem
    # solve_single_problem(problem_file="mknapcb1", problem_num=1)
