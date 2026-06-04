"""
OR-Library Utility Tool
Download and manage multidimensional knapsack test datasets from OR-Library.

Reference: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html
"""

import urllib.request
import os
from knapsack import parse_or_library_format

# Available OR-Library datasets for multidimensional knapsack
OR_LIBRARY_DATASETS = {
    # Traditional format files
    'mknap1': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap1.txt',
        'problems': 7,
        'items_range': '4-25',
        'constraints_range': '2-4',
        'description': 'Classic Petersen (1967) problems'
    },
    'mknap2': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2.txt',
        'problems': 48,
        'items_range': '6-105',
        'constraints_range': '2-30',
        'description': 'Collection from literature'
    },
    
    # Chu & Beasley (1998) format files
    'mknapcb1': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb2': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb2.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb3': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb3.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb4': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb4.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb5': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb5.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb6': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb6.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb7': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb7.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb8': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb8.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
    'mknapcb9': {
        'url': 'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb9.txt',
        'problems': 30,
        'items': 100,
        'constraints': 5,
        'description': 'Chu & Beasley 100 items, 5 constraints'
    },
}


def list_datasets():
    """Display available OR-Library datasets."""
    print("\n" + "="*80)
    print("AVAILABLE OR-LIBRARY MULTIDIMENSIONAL KNAPSACK DATASETS")
    print("="*80 + "\n")
    
    print("TRADITIONAL FORMAT (Variable sizes):")
    print("-" * 80)
    for name, info in list(OR_LIBRARY_DATASETS.items())[:2]:
        print(f"  {name:12s} - {info['description']}")
        print(f"                {info['problems']} problems | Items: {info['items_range']} | Constraints: {info['constraints_range']}")
        print()
    
    print("\nCHU & BEASLEY FORMAT (100 items, 5 constraints):")
    print("-" * 80)
    print("  Each file contains 30 problems with different tightness ratios:")
    print("    - Problems 1-10:  Tightness 0.25 (loose constraints)")
    print("    - Problems 11-20: Tightness 0.50 (medium constraints)")
    print("    - Problems 21-30: Tightness 0.75 (tight constraints)")
    print()
    for name, info in list(OR_LIBRARY_DATASETS.items())[2:]:
        print(f"  {name:12s} - {info['description']}")
    
    print("\n" + "="*80)


def download_dataset(dataset_name, save_local=False):
    """
    Download a dataset from OR-Library.
    
    Parameters:
    - dataset_name: name of the dataset (e.g., 'mknapcb1')
    - save_local: if True, save to local file
    
    Returns:
    - List of parsed problems or None if download fails
    """
    
    if dataset_name not in OR_LIBRARY_DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(OR_LIBRARY_DATASETS.keys())}")
        return None
    
    info = OR_LIBRARY_DATASETS[dataset_name]
    url = info['url']
    
    print(f"\nDownloading {dataset_name}...")
    print(f"Source: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        
        # Save locally if requested
        if save_local:
            filename = f"{dataset_name}.txt"
            with open(filename, 'w') as f:
                f.write(data)
            print(f"Saved to: {filename}")
        
        # Parse and return
        problems = parse_or_library_format(data)
        print(f"Successfully loaded {len(problems)} problems.\n")
        return problems
    
    except Exception as e:
        print(f"Error downloading dataset: {e}\n")
        return None


def get_problem_info(dataset_name, problem_num=None):
    """
    Get information about a specific problem in a dataset.
    
    Parameters:
    - dataset_name: name of the dataset
    - problem_num: problem number (1-indexed), or None for all
    """
    
    problems = download_dataset(dataset_name, save_local=False)
    if not problems:
        return None
    
    if problem_num is None:
        # Show all problems
        print(f"\n{dataset_name.upper()} - Problem Summary")
        print("="*80)
        for p in problems:
            print(f"Problem {p['problem_id']:2d}: {p['num_items']:3d} items × "
                  f"{p['num_constraints']} constraints | "
                  f"Optimal: {p['optimal_value'] if p['optimal_value'] > 0 else 'Unknown':>6}")
    else:
        # Show specific problem
        if 1 <= problem_num <= len(problems):
            p = problems[problem_num - 1]
            print(f"\n{dataset_name.upper()} - Problem {problem_num}")
            print("="*80)
            print(f"  Items: {p['num_items']}")
            print(f"  Constraints: {p['num_constraints']}")
            print(f"  Optimal value: {p['optimal_value'] if p['optimal_value'] > 0 else 'Unknown'}")
            print(f"  Item values (first 20): {p['values'][:20]}")
            print(f"  Capacity limits: {p['capacities']}")
            return p
        else:
            print(f"Error: Problem {problem_num} not found in {dataset_name}")
            return None


def compare_datasets():
    """Compare characteristics of all available datasets."""
    print("\n" + "="*80)
    print("OR-LIBRARY DATASETS COMPARISON")
    print("="*80 + "\n")
    
    print(f"{'Dataset':15s} {'Problems':>10s} {'Items':>12s} {'Constraints':>14s} {'Description':30s}")
    print("-" * 80)
    
    for name, info in OR_LIBRARY_DATASETS.items():
        if 'items_range' in info:
            items = info['items_range']
            constraints = info['constraints_range']
        else:
            items = str(info['items'])
            constraints = str(info['constraints'])
        
        problems = str(info['problems'])
        desc = info['description'][:28]
        
        print(f"{name:15s} {problems:>10s} {items:>12s} {constraints:>14s}  {desc:28s}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OR-LIBRARY DATASET UTILITY")
    print("="*80)
    
    # List all available datasets
    list_datasets()
    
    # Show dataset comparison
    compare_datasets()
    
    # Example: Download and analyze mknapcb1
    print("\nExample: Loading mknapcb1 dataset...")
    print("-" * 80)
    
    problems = download_dataset("mknapcb1", save_local=False)
    
    if problems:
        get_problem_info("mknapcb1")
        
        print("\nTightness Categories in mknapcb Files:")
        print("-" * 80)
        print("The mknapcb (Chu & Beasley) files have 30 problems each categorized by")
        print("constraint tightness ratio (ratio of sum of weights to capacity):")
        print()
        print("  Loose   (0.25): Problems  1-10  - Easier to solve")
        print("  Medium  (0.50): Problems 11-20  - Medium difficulty")
        print("  Tight   (0.75): Problems 21-30  - Harder to solve")
        print()
        print("You can use these in solve_or_library.py:")
        print()
        print("  from solve_or_library import solve_multiple_problems")
        print("  solve_multiple_problems('mknapcb1', num_to_solve=10)")
