"""
Multidimensional Knapsack Problem Solver (OR-Library)
Optimized single-file implementation using Genetic Algorithm (PyGAD)
with Greedy Repair and Greedy Initialization.

Usage examples:
    python mkp_solver.py                              # default: solve 3 problems from mknapcb1
    python mkp_solver.py --file mknapcb1 --num 5      # solve first 5 problems
    python mkp_solver.py --file mknapcb1 --single 1   # solve only problem #1
    python mkp_solver.py --gens 200 --pop 80          # tune GA parameters
"""

import argparse
import os
import urllib.request
import traceback
import numpy as np
import pygad
import pandas as pd

OR_LIBRARY_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}.txt"


# ============================================================================
# Data Loading
# ============================================================================

def load_or_library(name="mknapcb1", use_cache=True, timeout=15):
    """Load OR-Library dataset (from local cache if available, else download)."""
    cache_file = f"{name}.txt"

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"[cache] loaded {cache_file}")
    else:
        url = OR_LIBRARY_URL.format(name=name)
        print(f"[download] {url}")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read().decode("utf-8")
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"[cache] saved to {cache_file}")

    return parse_or_library(data)


def parse_or_library(text):
    """Parse OR-Library multidimensional knapsack file format."""
    t = list(map(int, text.split()))
    i = 0
    num_problems = t[i]; i += 1
    problems = []
    for p in range(num_problems):
        n, m, opt = t[i], t[i + 1], t[i + 2]
        i += 3
        values = t[i:i + n]; i += n
        weights = [t[i + k * n:i + (k + 1) * n] for k in range(m)]
        i += m * n
        capacities = t[i:i + m]; i += m
        problems.append({
            "id": p + 1, "n": n, "m": m, "optimal": opt,
            "values": values, "weights": weights, "capacities": capacities,
        })
    return problems


# ============================================================================
# Knapsack Problem
# ============================================================================

class MKP:
    """Multidimensional Knapsack Problem with vectorized operations."""

    def __init__(self, values, weights, capacities, optimal=0, pid=None):
        self.values = np.asarray(values, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)     # shape (m, n)
        self.capacities = np.asarray(capacities, dtype=np.float64)  # shape (m,)
        self.n = len(values)
        self.m = len(capacities)
        self.optimal = optimal
        self.pid = pid

        # Precompute pseudo-utility ratio for greedy operations:
        # sum of normalized weight across dimensions vs value
        weight_sum = self.weights.sum(axis=0)  # shape (n,)
        weight_sum = np.where(weight_sum == 0, 1e-9, weight_sum)
        self.ratio = self.values / weight_sum
        self.order_desc = np.argsort(-self.ratio)  # best first
        self.order_asc = np.argsort(self.ratio)    # worst first

    # -- evaluation ----------------------------------------------------------
    def total_value(self, sol):
        return float(np.dot(sol, self.values))

    def total_weights(self, sol):
        return self.weights @ sol

    def is_feasible(self, sol):
        return np.all(self.total_weights(sol) <= self.capacities)

    # -- repair / construction ----------------------------------------------
    def greedy_repair(self, sol):
        """If infeasible, drop low-ratio items until feasible; then add others."""
        sol = sol.astype(np.int8).copy()
        loads = self.total_weights(sol)

        # DROP phase: remove worst items while infeasible
        for idx in self.order_asc:
            if np.all(loads <= self.capacities):
                break
            if sol[idx] == 1:
                sol[idx] = 0
                loads -= self.weights[:, idx]

        # ADD phase: try to add best items that still fit
        for idx in self.order_desc:
            if sol[idx] == 0:
                new_loads = loads + self.weights[:, idx]
                if np.all(new_loads <= self.capacities):
                    sol[idx] = 1
                    loads = new_loads
        return sol

    def greedy_solution(self):
        """Pure greedy construction (used to seed the GA population)."""
        sol = np.zeros(self.n, dtype=np.int8)
        loads = np.zeros(self.m)
        for idx in self.order_desc:
            new_loads = loads + self.weights[:, idx]
            if np.all(new_loads <= self.capacities):
                sol[idx] = 1
                loads = new_loads
        return sol


# ============================================================================
# Genetic Algorithm Solver
# ============================================================================

def solve_mkp(problem, generations=100, pop_size=50, seed=42,
              gap_threshold=0.001, stagnation=40, verbose=True):
    """
    Solve MKP using PyGAD with greedy repair + greedy seeding.
    Returns (best_solution, total_value, gap_percent_or_None).
    """
    n = problem.n

    # ---- Build initial population: 1 greedy + rest random (then repaired) --
    rng = np.random.default_rng(seed)
    init_pop = np.zeros((pop_size, n), dtype=np.int8)
    init_pop[0] = problem.greedy_solution()
    for i in range(1, pop_size):
        # Random density biased toward feasibility (~30-60% items)
        density = rng.uniform(0.3, 0.6)
        sol = (rng.random(n) < density).astype(np.int8)
        init_pop[i] = problem.greedy_repair(sol)

    # ---- Fitness: repair the solution then return its value ---------------
    # We store repaired versions back to ga_instance via on_generation hook.
    stats = {"best": -np.inf, "stagn": 0, "best_sol": init_pop[0].copy()}

    def fitness_func(ga, sol, idx):
        sol = problem.greedy_repair(np.asarray(sol, dtype=np.int8))
        return problem.total_value(sol)

    def on_gen(ga):
        sol, fit, _ = ga.best_solution()
        sol_repaired = problem.greedy_repair(np.asarray(sol, dtype=np.int8))
        fit = problem.total_value(sol_repaired)

        if fit > stats["best"] + 1e-9:
            stats["best"] = fit
            stats["best_sol"] = sol_repaired
            stats["stagn"] = 0
        else:
            stats["stagn"] += 1

        if verbose and (ga.generations_completed % 20 == 0 or ga.generations_completed == 1):
            gap_str = ""
            if problem.optimal > 0:
                gap = (problem.optimal - stats["best"]) / problem.optimal * 100
                gap_str = f" | gap={gap:.2f}%"
            print(f"  gen {ga.generations_completed:>4} | best={stats['best']:.0f}{gap_str}")

        # Early stop conditions
        if problem.optimal > 0:
            gap = (problem.optimal - stats["best"]) / problem.optimal
            if gap <= gap_threshold:
                return "stop"
        if stats["stagn"] >= stagnation:
            if verbose:
                print(f"  [early stop: stagnation @ gen {ga.generations_completed}]")
            return "stop"

    ga = pygad.GA(
        num_generations=generations,
        num_parents_mating=max(2, pop_size // 2),
        fitness_func=fitness_func,
        initial_population=init_pop,
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        crossover_probability=0.9,
        mutation_type="random",
        mutation_probability=max(0.01, 1.0 / n),
        keep_elitism=2,
        random_seed=seed,
        on_generation=on_gen,
        suppress_warnings=True,
    )
    ga.run()

    best_sol = stats["best_sol"]
    best_val = problem.total_value(best_sol)
    gap = None
    if problem.optimal > 0:
        gap = (problem.optimal - best_val) / problem.optimal * 100
    return best_sol, best_val, gap


# ============================================================================
# Driver / CLI
# ============================================================================

def solve_batch(problem_file="mknapcb1", num=3, generations=100, pop_size=50,
                threshold_pct=1.0, verbose=True):
    """Solve several problems and report a summary."""
    print(f"\n{'='*70}\nMKP SOLVER — {problem_file}\n{'='*70}")
    problems = load_or_library(problem_file)
    if not problems:
        return None

    if num:
        problems = problems[:num]
    print(f"Solving {len(problems)} problem(s)...\n")

    rows = []
    for pdata in problems:
        pid = pdata["id"]
        print(f"\n── Problem {pid:2d} | items={pdata['n']} | "
              f"constraints={pdata['m']} | optimal={pdata['optimal']} ──")
        try:
            problem = MKP(pdata["values"], pdata["weights"], pdata["capacities"],
                          optimal=pdata["optimal"], pid=pid)
            sol, val, gap = solve_mkp(problem, generations=generations,
                                      pop_size=pop_size, verbose=verbose)
            selected = int(sol.sum())
            rows.append({
                "Problem": pid,
                "Items": pdata["n"],
                "Constraints": pdata["m"],
                "Optimal": pdata["optimal"] if pdata["optimal"] > 0 else None,
                "Found": int(val),
                "Selected": selected,
                "Gap (%)": round(gap, 2) if gap is not None else None,
                "Within {0}%?".format(threshold_pct):
                    "✓" if (gap is not None and gap <= threshold_pct) else "✗",
            })
            print(f"  → value={int(val)}  selected={selected}/{pdata['n']}"
                  + (f"  gap={gap:.2f}%" if gap is not None else ""))
        except Exception as e:
            print(f"  ERROR on problem {pid}: {e}")
            traceback.print_exc()

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    if not rows:
        print("No results.")
        return rows

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    gaps = [r["Gap (%)"] for r in rows if r["Gap (%)"] is not None]
    if gaps:
        print(f"\nStatistics over {len(gaps)} problems:")
        print(f"  average gap : {np.mean(gaps):.2f}%")
        print(f"  best  gap   : {min(gaps):.2f}%")
        print(f"  worst gap   : {max(gaps):.2f}%")
        within = sum(1 for g in gaps if g <= threshold_pct)
        print(f"  within {threshold_pct}% of optimal: {within}/{len(gaps)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="OR-Library MKP Solver")
    ap.add_argument("--file", default="mknapcb1",
                    help="OR-Library dataset name (mknap1, mknap2, mknapcb1..9)")
    ap.add_argument("--num", type=int, default=3,
                    help="how many problems to solve (default 3, 0 = all)")
    ap.add_argument("--single", type=int, default=None,
                    help="solve only this single problem id (1-indexed)")
    ap.add_argument("--gens", type=int, default=100, help="GA generations")
    ap.add_argument("--pop", type=int, default=50, help="GA population size")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="percent gap considered 'optimal' (default 1.0)")
    ap.add_argument("--quiet", action="store_true", help="less per-gen output")
    args = ap.parse_args()

    if args.single is not None:
        problems = load_or_library(args.file)
        if not problems or args.single < 1 or args.single > len(problems):
            print(f"Problem #{args.single} not found in {args.file}.")
            return
        pdata = problems[args.single - 1]
        print(f"\nSolving problem {args.single} of {args.file}...")
        problem = MKP(pdata["values"], pdata["weights"], pdata["capacities"],
                      optimal=pdata["optimal"], pid=pdata["id"])
        sol, val, gap = solve_mkp(problem, generations=args.gens,
                                  pop_size=args.pop, verbose=not args.quiet)
        print(f"\nValue   : {int(val)}")
        print(f"Selected: {int(sol.sum())}/{pdata['n']}")
        if gap is not None:
            print(f"Gap     : {gap:.2f}%")
        return

    solve_batch(
        problem_file=args.file,
        num=args.num if args.num > 0 else None,
        generations=args.gens,
        pop_size=args.pop,
        threshold_pct=args.threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
