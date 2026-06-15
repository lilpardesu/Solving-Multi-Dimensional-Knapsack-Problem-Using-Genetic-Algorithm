"""
Multidimensional Knapsack Problem Solver — fully local & auto-tuned.

Place OR-Library dataset files (e.g. mknapcb1.txt) in ./data/

Usage:
    python mkp_solver.py                    # solves problem #1 of mknapcb1
    python mkp_solver.py --file mknapcb2    # different dataset
    python mkp_solver.py --single 7         # different problem
    python mkp_solver.py --all              # solve all problems in the file
    python mkp_solver.py --list             # list available local datasets
"""

import argparse
import os
import sys
import traceback
import numpy as np
import pygad
import pandas as pd

DATA_DIR = "data"


# ============================================================================
# Data Loading
# ============================================================================

def load_or_library(name="mknapcb1", data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{name}.txt")
    if not os.path.exists(path):
        print(f"\nERROR: file not found: {path}")
        print(f"Place '{name}.txt' inside the '{data_dir}/' folder.\n")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"[local] loaded {path}")
    return parse_or_library(data)


def parse_or_library(text):
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
# Known-best & LP results (mkcbres.txt)
# ============================================================================

# mknapcb file → (m, n) mapping
MKNAPCB_MAP = {
    "mknapcb1": (5, 100),  "mknapcb2": (5, 250),  "mknapcb3": (5, 500),
    "mknapcb4": (10, 100), "mknapcb5": (10, 250), "mknapcb6": (10, 500),
    "mknapcb7": (30, 100), "mknapcb8": (30, 250), "mknapcb9": (30, 500),
}


def load_known_results(data_dir=DATA_DIR, filename="mkcbres.txt"):
    """
    Parse mkcbres.txt. Returns {(m, n, idx): {"bks": int, "lp": float}}.
    idx is 0-based as in the file (00..29).
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return {}

    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            name, val = parts
            # name looks like "5.100-00"
            if "." not in name or "-" not in name:
                continue
            try:
                left, idx = name.split("-")
                m, n = left.split(".")
                m, n, idx = int(m), int(n), int(idx)
                v = float(val)
            except ValueError:
                continue

            key = (m, n, idx)
            entry = results.setdefault(key, {})
            # BKS appears first (integer), LP appears later (scientific)
            if "bks" not in entry:
                entry["bks"] = int(v)
            else:
                entry["lp"] = v
    return results


def lookup_known(file_name, problem_id, known_results):
    """Look up BKS and LP for a given mknapcb file + 1-based problem id."""
    if file_name not in MKNAPCB_MAP:
        return None, None
    m, n = MKNAPCB_MAP[file_name]
    entry = known_results.get((m, n, problem_id - 1))
    if not entry:
        return None, None
    return entry.get("bks"), entry.get("lp")
 

def list_local_datasets(data_dir=DATA_DIR):
    if not os.path.isdir(data_dir):
        print(f"No '{data_dir}/' folder found.")
        return []
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if not files:
        print(f"No .txt files in '{data_dir}/'.")
        return []
    print(f"\nLocal datasets in '{data_dir}/':")
    for f in files:
        print(f"  - {f}")
    return files


# ============================================================================
# Auto-Tuning — choose GA parameters from problem dimensions
# ============================================================================

def auto_tune(n, m):
    """
    Pick GA hyper-parameters based on problem size.
    Returns a dict of recommended settings.
    """
    # --- population & generations scale with n ------------------------------
    if n <= 50:
        pop_size, generations, stagnation = 40, 150, 30
        size_tag = "small"
    elif n <= 100:
        pop_size, generations, stagnation = 80, 300, 50
        size_tag = "medium"
    elif n <= 250:
        pop_size, generations, stagnation = 150, 600, 80
        size_tag = "large"
    else:
        pop_size, generations, stagnation = 200, 1000, 120
        size_tag = "x-large"

    # --- more constraints → harder problem → bump effort up -----------------
    if m >= 10:
        pop_size = int(pop_size * 1.3)
        generations = int(generations * 1.3)
        stagnation = int(stagnation * 1.3)
    elif m >= 5:
        pop_size = int(pop_size * 1.1)
        generations = int(generations * 1.1)

    # --- mutation rate: standard 1/n, slightly higher for tight problems ----
    mutation_rate = max(0.01, 1.0 / n)
    if m / max(n, 1) > 0.1:           # many constraints relative to items
        mutation_rate *= 1.5

    # --- crossover & elitism are robust defaults ----------------------------
    return {
        "size_tag": size_tag,
        "pop_size": pop_size,
        "generations": generations,
        "stagnation": stagnation,
        "mutation_rate": round(mutation_rate, 4),
        "crossover_prob": 0.9,
        "elitism": max(2, pop_size // 25),
        "tournament_k": 3,
        "gap_threshold": 0.001,   # stop if within 0.1% of optimum
    }


# ============================================================================
# Knapsack Problem
# ============================================================================

class MKP:
    def __init__(self, values, weights, capacities, optimal=0, pid=None):
        self.values = np.asarray(values, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.capacities = np.asarray(capacities, dtype=np.float64)
        self.n = len(values)
        self.m = len(capacities)
        self.optimal = optimal
        self.pid = pid

        weight_sum = self.weights.sum(axis=0)
        weight_sum = np.where(weight_sum == 0, 1e-9, weight_sum)
        self.ratio = self.values / weight_sum
        self.order_desc = np.argsort(-self.ratio)
        self.order_asc = np.argsort(self.ratio)

    def total_value(self, sol):
        return float(np.dot(sol, self.values))

    def total_weights(self, sol):
        return self.weights @ sol

    def greedy_repair(self, sol):
        sol = sol.astype(np.int8).copy()
        loads = self.total_weights(sol)
        for idx in self.order_asc:
            if np.all(loads <= self.capacities):
                break
            if sol[idx] == 1:
                sol[idx] = 0
                loads -= self.weights[:, idx]
        for idx in self.order_desc:
            if sol[idx] == 0:
                new_loads = loads + self.weights[:, idx]
                if np.all(new_loads <= self.capacities):
                    sol[idx] = 1
                    loads = new_loads
        return sol

    def greedy_solution(self):
        sol = np.zeros(self.n, dtype=np.int8)
        loads = np.zeros(self.m)
        for idx in self.order_desc:
            new_loads = loads + self.weights[:, idx]
            if np.all(new_loads <= self.capacities):
                sol[idx] = 1
                loads = new_loads
        return sol


# ============================================================================
# Genetic Algorithm (uses auto-tuned params)
# ============================================================================

def solve_mkp(problem, params=None, seed=42, verbose=True):
    if params is None:
        params = auto_tune(problem.n, problem.m)

    n = problem.n
    pop_size = params["pop_size"]
    rng = np.random.default_rng(seed)

    init_pop = np.zeros((pop_size, n), dtype=np.int8)
    init_pop[0] = problem.greedy_solution()
    for i in range(1, pop_size):
        density = rng.uniform(0.3, 0.6)
        sol = (rng.random(n) < density).astype(np.int8)
        init_pop[i] = problem.greedy_repair(sol)

    stats = {"best": -np.inf, "stagn": 0, "best_sol": init_pop[0].copy()}

    def fitness_func(ga, sol, idx):
        sol = problem.greedy_repair(np.asarray(sol, dtype=np.int8))
        return problem.total_value(sol)

    def on_gen(ga):
        sol, _, _ = ga.best_solution()
        sol_rep = problem.greedy_repair(np.asarray(sol, dtype=np.int8))
        fit = problem.total_value(sol_rep)

        if fit > stats["best"] + 1e-9:
            stats["best"] = fit
            stats["best_sol"] = sol_rep
            stats["stagn"] = 0
        else:
            stats["stagn"] += 1

        gen = ga.generations_completed
        if verbose and (gen % 25 == 0 or gen == 1):
            extra = ""
            if problem.optimal > 0:
                gap = (problem.optimal - stats["best"]) / problem.optimal * 100
                extra = f" | gap={gap:.2f}%"
            print(f"  gen {gen:>4} | best={stats['best']:.0f}{extra}")

        if problem.optimal > 0:
            gap = (problem.optimal - stats["best"]) / problem.optimal
            if gap <= params["gap_threshold"]:
                if verbose:
                    print(f"  [early stop: reached target gap @ gen {gen}]")
                return "stop"
        if stats["stagn"] >= params["stagnation"]:
            if verbose:
                print(f"  [early stop: stagnation @ gen {gen}]")
            return "stop"

    ga = pygad.GA(
        num_generations=params["generations"],
        num_parents_mating=max(2, pop_size // 2),
        fitness_func=fitness_func,
        initial_population=init_pop,
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="tournament",
        K_tournament=params["tournament_k"],
        crossover_type="uniform",
        crossover_probability=params["crossover_prob"],
        mutation_type="random",
        mutation_probability=params["mutation_rate"],
        keep_elitism=params["elitism"],
        random_seed=seed,
        on_generation=on_gen,
        suppress_warnings=True,
    )
    ga.run()

    best_sol = stats["best_sol"]
    best_val = problem.total_value(best_sol)
    gap = (problem.optimal - best_val) / problem.optimal * 100 if problem.optimal > 0 else None
    return best_sol, best_val, gap


# ============================================================================
# Driver
# ============================================================================

def print_params(params, n, m):
    print(f"  auto-tuned params  → size class : {params['size_tag']}  (n={n}, m={m})")
    print(f"     population      = {params['pop_size']}")
    print(f"     generations     = {params['generations']}")
    print(f"     stagnation stop = {params['stagn'] if 'stagn' in params else params['stagnation']}")
    print(f"     mutation rate   = {params['mutation_rate']}")
    print(f"     crossover prob  = {params['crossover_prob']}")
    print(f"     elitism         = {params['elitism']}")
    print(f"     gap target      = {params['gap_threshold']*100:.2f}%")

def solve_single(problem_file, pid, verbose=True):
    print(f"\n{'='*70}\nMKP SOLVER — {problem_file}  | problem #{pid}\n{'='*70}")
    problems = load_or_library(problem_file)
    if pid < 1 or pid > len(problems):
        print(f"Problem #{pid} not in {problem_file} (has {len(problems)} problems).")
        return None
    pdata = problems[pid - 1]

    # Lookup known best solution & LP from mkcbres.txt
    known = load_known_results()
    bks, lp = lookup_known(problem_file, pid, known)

    problem = MKP(pdata["values"], pdata["weights"], pdata["capacities"],
                  optimal=0, pid=pdata["id"])
    params = auto_tune(problem.n, problem.m)

    print(f"\nProblem {pid}: items={problem.n}, constraints={problem.m}")
    if bks is not None:
        print(f"  Known best solution (Chu & Beasley 1997): {bks}")
    if lp is not None:
        print(f"  LP relaxation (upper bound)             : {lp:.2f}")
    print_params(params, problem.n, problem.m)
    print()

    sol, val, _ = solve_mkp(problem, params=params, verbose=verbose)

    print(f"\n{'─'*70}\nRESULT")
    print(f"  value            : {int(val)}")
    print(f"  selected         : {int(sol.sum())}/{problem.n} items")
    if bks is not None:
        gap_bks = (bks - val) / bks * 100
        sign = "+" if val > bks else ""
        diff = int(val - bks)
        print(f"  vs BKS ({bks})   : {sign}{diff}  ({gap_bks:+.2f}% gap)")
        if val >= bks:
            print(f"  🏆 matched or beat the known best!")
        elif gap_bks < 1.0:
            print(f"  ✓ within 1% of best known — excellent")
        elif gap_bks < 3.0:
            print(f"  ✓ within 3% of best known — good")
    if lp is not None:
        gap_lp = (lp - val) / lp * 100
        print(f"  vs LP  ({lp:.0f}) : {gap_lp:.2f}% gap (LP is an upper bound)")
    return sol, val


def solve_all(problem_file, verbose=False):
    print(f"\n{'='*70}\nMKP SOLVER — {problem_file} (ALL)\n{'='*70}")
    problems = load_or_library(problem_file)
    known = load_known_results()
    rows = []
    for pdata in problems:
        pid = pdata["id"]
        bks, lp = lookup_known(problem_file, pid, known)
        print(f"\n── Problem {pid:2d} | n={pdata['n']} m={pdata['m']}"
              + (f" | BKS={bks}" if bks else "") + " ──")
        try:
            problem = MKP(pdata["values"], pdata["weights"], pdata["capacities"],
                          optimal=0, pid=pid)
            params = auto_tune(problem.n, problem.m)
            sol, val, _ = solve_mkp(problem, params=params, verbose=verbose)
            gap_bks = (bks - val) / bks * 100 if bks else None
            rows.append({
                "Problem": pid,
                "n": pdata["n"], "m": pdata["m"],
                "BKS": bks,
                "Found": int(val),
                "Gap vs BKS (%)": round(gap_bks, 2) if gap_bks is not None else None,
                "Selected": int(sol.sum()),
            })
            tag = f" gap={gap_bks:.2f}%" if gap_bks is not None else ""
            print(f"  → value={int(val)}{tag}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    gaps = [r["Gap vs BKS (%)"] for r in rows if r["Gap vs BKS (%)"] is not None]
    if gaps:
        print(f"\naverage gap vs BKS : {np.mean(gaps):.2f}%")
        print(f"best  gap          : {min(gaps):.2f}%")
        print(f"worst gap          : {max(gaps):.2f}%")
        print(f"within 1% of BKS   : {sum(1 for g in gaps if g <= 1.0)}/{len(gaps)}")
        print(f"matched or beat BKS: {sum(1 for g in gaps if g <= 0)}/{len(gaps)}")


def main():
    ap = argparse.ArgumentParser(description="OR-Library MKP Solver (local, auto-tuned)")
    ap.add_argument("--file", default="mknapcb1",
                    help="dataset name (without .txt) inside ./data/")
    ap.add_argument("--single", type=int, default=1,
                    help="solve this problem id (default 1)")
    ap.add_argument("--all", action="store_true",
                    help="solve all problems in the file")
    ap.add_argument("--list", action="store_true",
                    help="list local datasets and exit")
    ap.add_argument("--quiet", action="store_true", help="less output")
    
    args = ap.parse_args()

    if args.list:
        list_local_datasets()
        return
    if args.all:
        solve_all(args.file, verbose=not args.quiet)
        return
    solve_single(args.file, args.single, verbose=not args.quiet)


if __name__ == "__main__":
    main()
