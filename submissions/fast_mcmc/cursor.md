# Cursor AI Prompt Context: HRT Macro Placement Challenge

## 1. Project Objective
Build a High-Speed Markov Chain Monte Carlo (MCMC) Simulated Annealing placement engine to solve the VLSI Macro Placement problem. 
- **Goal:** Minimize the Proxy Cost: `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`.
- **Hard Constraint:** Strictly ZERO overlaps between hard macros.
- **Time Constraint:** Max 1 hour of execution time per benchmark (AMD EPYC 16-core).

## 2. Repository Rules (Strict Constraints)
We are building a submission for an existing evaluation repository. 

**DO NOT TOUCH THESE FILES/FOLDERS:**
- `macro_place/` (The core TILOS evaluation logic)
- `external/` (The C++ MacroPlacement submodule)
- `evaluate` / `evaluate.py` (The main runner script)
- `benchmarks/` 
- `requirements.txt` / `pyproject.toml` (Unless adding `numba`, but do not modify existing deps)

**WHERE OUR CODE GOES:**
All of our code must be isolated inside a new directory within the submissions folder:
`submissions/fast_mcmc/`

When the orchestrator runs, it will be executed from the root directory via:
`uv run evaluate submissions/fast_mcmc/main.py -b ibm01`

## 3. Directory Structure to Build
Inside `submissions/fast_mcmc/`, build the following Data-Oriented structure:

```text
submissions/fast_mcmc/
├── main.py              # Orchestrator: parses benchmark, launches multiprocessing Pool, collects results
├── worker.py            # The isolated MCMC loop (runs on a single core)
├── initialization.py    # GRASP (Greedy Randomized Adaptive Search Procedure) initial placement
├── fast_eval.py         # 100% pure functions decorated with @njit (Numba) for O(1) deltas
└── state.py             # Memory definitions (NumPy arrays, NO complex OOP classes)
```

## 4. Architectural Directives

### A. Data-Oriented Design (No Heavy OOP)
- Do NOT build `class Macro:` or `class Net:`. 
- Represent the board state entirely as contiguous arrays in `state.py`.
- **MacroCoords:** `(N, 2)` NumPy array for X, Y coordinates.
- **Adjacency List:** Pre-computed 1D/2D integer arrays mapping `macro_id -> nets` and `net_id -> macros`.
- **Spatial Grid:** A 2D integer array matrix binning the continuous canvas to allow O(1) localized overlap lookups.

### B. Fast Evaluator (The Bottleneck)
- All logic inside `fast_eval.py` must be compiled to C using Numba (`@njit`).
- **Delta Wirelength:** When a macro moves, do not recalculate the whole board. Look up its connected nets via the Adjacency List and only recalculate their Half-Perimeter Wirelength (HPWL).
- **Delta Overlap:** When a macro moves, only check for collisions within the specific bins of the `Spatial Grid` it just entered or exited.

### C. Multiprocessing Strategy
- Python's Global Interpreter Lock (GIL) is a blocker. Use `multiprocessing.Pool` in `main.py`.
- **Main Thread:** Reads benchmark, builds the Adjacency List, forks 16 workers, passing the Adjacency List and a unique `seed` to each.
- **Worker Thread:** 1. Immediately runs `initialization.py` (GRASP) using its unique seed so all 16 workers start in highly-optimized but mathematically distinct states.
  2. Drops into the MCMC Simulated Annealing loop (`while temp > 0:`).
  3. Uses a biased perturbation strategy (weights probability of moving macros with the worst current wirelength/overlap penalties).
  4. Returns the final valid placement coordinates right before the 60-minute timeout.
- **Main Thread:** Collects the 16 results, evaluates the final proxy scores, and returns the absolute best one.

## 5. Implementation Sequence
1. **Parser & State:** Write `state.py` to ingest the benchmark data provided by the evaluator and convert it into Numpy arrays and the Adjacency List.
2. **Numba Deltas:** Write the math in `fast_eval.py` and verify Numba compiles it successfully.
3. **GRASP Setup:** Write `initialization.py` to place macros semi-greedily onto the Spatial Grid.
4. **MCMC Loop:** Write `worker.py` to execute the Metropolis-Hastings acceptance logic.
5. **Parallelization:** Wrap it all in `main.py` using `multiprocessing`.