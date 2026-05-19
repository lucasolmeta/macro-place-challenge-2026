# Cursor AI Prompt Context: HRT Macro Placement Challenge

## 1. Project Objective
Build a High-Speed Markov Chain Monte Carlo (MCMC) Simulated Annealing placement engine to solve the VLSI Macro Placement problem. 
- **Goal:** Minimize the Proxy Cost: `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`.
- **Hard Constraint:** Strictly ZERO overlaps between hard macros at termination.
- **Testing Flex:** Must support small-scale local testing via CLI overrides (custom runtime, specified benchmarks, and variable worker counts).

## 2. Repository Rules (Strict Constraints)
We are building a submission for an existing evaluation repository. 

**DO NOT TOUCH THESE FILES/FOLDERS:**
- `macro_place/` (The core TILOS evaluation logic)
- `external/` (The C++ MacroPlacement submodule)
- `evaluate` / `evaluate.py` (The main runner script)
- `benchmarks/` 
- `requirements.txt` / `pyproject.toml` (Unless adding `numba`, do not modify existing deps)

**WHERE OUR CODE GOES:**
All of our code must be isolated inside a new directory within the submissions folder:
`submissions/fast_mcmc/`

When the orchestrator runs, it will be executed from the root directory via:
`uv run evaluate submissions/fast_mcmc/main.py -b ibm01`

## 3. Cursor Scaffolding Instructions
Cursor, your first task is to generate the broad codebase structure listed below. 
- Create all files, classes, and complete function signatures.
- Write extensive docstrings and inner code documentation detailing the mathematical intent.
- Implement the baseline control flow and structural plumbing completely.
- For the dense algorithmic logic (e.g., exact Numba delta math or the GRASP packing loop), provide partial boilerplate along with explicit `# TODO: [Step-by-Step Instructions]` comments guiding the user to complete it manually.

```text
submissions/fast_mcmc/
├── main.py              # Entry/CLI: parses args, configures testing limits, manages Pool
├── worker.py            # MCMC Loop: manages annealing schedules and Metropolis logic
├── initialization.py    # Constructive Setup: GRASP placement routines
├── fast_eval.py         # Performance Math: Numba-compiled (@njit) exact delta calculations
└── state.py             # Data-Oriented Memory: NumPy array shapes and grid states
```

## 4. Architectural Directives

### A. Data-Oriented Design (No Heavy OOP)
- Do NOT build a `class Macro` or `class Net`. Attributes introduce lookup overhead.
- Represent the board state entirely as contiguous arrays in `state.py`.
- **MacroCoords:** `(N, 2)` NumPy array for X, Y coordinates.
- **Adjacency List:** Pre-computed 1D/2D integer arrays mapping `macro_id -> nets` and `net_id -> macros`.
- **Spatial Grid:** A 2D integer array matrix binning the continuous canvas to allow localized overlap and density lookups.

### B. Multi-Tier Overlap Prevention
To guarantee zero overlaps without completely freezing early exploration:
1. **Annealing Phase:** The Spatial Grid tracks overlap counts. Overlaps are permitted but penalized heavily in the cost function, scaling up exponentially as temperature decays.
2. **Hard Filter:** Any perturbation proposal that results in a hard macro completely eclipsing another or breaching the canvas boundary is rejected *prior* to proxy cost evaluation.
3. **Legalization Phase:** In the final temperature stages, transition the engine to a strict zero-overlap acceptance mode (any move causing a new overlap is immediately discarded).

### C. High-Precision Proxy Cost Delta Heuristic
Because proxy cost contains non-linear density and congestion elements, the delta evaluator in `fast_eval.py` must be highly accurate:
- **Wirelength Delta:** Track bounding boxes `[min_x, max_x, min_y, max_y]` for every net. When a macro moves, use the adjacency list to recompute the half-perimeter wirelength (HPWL) change for *only* its connected nets.
- **Density/Congestion Delta:** Instead of querying the entire layout, calculate the local change in the 2D Spatial Grid matrix. Compute the structural difference in cell density *only* within the localized grid cells that the perturbed macro exited and entered.

### D. Scalable Testing Harness
Modify `main.py` to parse custom command-line arguments to allow isolated debugging:
- `--benchmark`: String specifying the target practice problem (e.g., `ibm01`).
- `--workers`: Integer capping the multiprocessing pool size (e.g., 1 to 16).
- `--timeout`: Integer setting a strict execution time limit in seconds to easily test 30-second or 5-minute diagnostic runs.

## 5. Implementation Sequence
1. **Parser & State:** Scaffold `state.py` to ingest benchmark parameters and set up NumPy matrices.
2. **Numba Deltas:** Scaffold the high-precision delta functions in `fast_eval.py` with `@njit` compilation decorators.
3. **GRASP Setup:** Scaffold the semi-greedy sequential placement in `initialization.py`.
4. **MCMC Loop:** Scaffold the core annealing engine and Metropolis-Hastings acceptance logic in `worker.py`.
5. **Parallelization:** Build the multiprocessing configuration layer in `main.py` ensuring CLI runtime parameters are strictly enforced.