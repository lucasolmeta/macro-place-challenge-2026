# Cursor AI Prompt Context: High-Performance MCMC Macro Placer

## 1. Project Objective
<<<<<<< HEAD
Build a High-Speed Markov Chain Monte Carlo (MCMC) Simulated Annealing placement engine to solve the VLSI Macro Placement problem. 
- **Goal:** Minimize the Proxy Cost: `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`.
- **Hard Constraint:** Strictly ZERO overlaps between hard macros at termination.
- **Testing Flex:** Must support small-scale local testing via CLI overrides (custom runtime, specified benchmarks, and variable worker counts).
=======
Build a fully functioning, high-speed, non-blocking Markov Chain Monte Carlo (MCMC) Simulated Annealing placement engine to solve the VLSI Macro Placement problem.
- **Objective Function:** Minimize the Proxy Cost: $1.0 \times \text{Wirelength} + 0.5 \times \text{Density} + 0.5 \times \text{Congestion}$.
- **Hard Constraints:** Strictly ZERO spatial overlaps between hard macros at termination and ZERO canvas boundary violations.
- **Performance Requirement:** Maximize execution throughput. Code must be capable of processing millions of state evaluations. Pairwise geometric loops ($O(N)$ or $O(N^2)$ checks per proposal) are strictly prohibited.
>>>>>>> attempt2

## 2. Repository Rules & Boundary Constraints
We are building a clean submission module for an existing evaluation harness. 

<<<<<<< HEAD
**DO NOT TOUCH THESE FILES/FOLDERS:**
- `macro_place/` (The core TILOS evaluation logic)
- `external/` (The C++ MacroPlacement submodule)
- `evaluate` / `evaluate.py` (The main runner script)
- `benchmarks/` 
- `requirements.txt` / `pyproject.toml` (Unless adding `numba`, do not modify existing deps)
=======
**STRICTLY PROHIBITED FROM MODIFING:**
- `macro_place/` (Core evaluation logic)
- `external/` (C++ submodules)
- `evaluate` / `evaluate.py` (Main runner framework)
- `benchmarks/` (Public benchmark datasets)
- Existing dependencies in `requirements.txt` or `pyproject.toml` (You may use `numba` and `numpy`)
>>>>>>> attempt2

**SUBMISSION LOCATION:**
All created code files must reside entirely within a single isolated directory:
`submissions/fast_mcmc/`

The orchestrator will execute our codebase from the project root using:
`uv run evaluate submissions/fast_mcmc/main.py -b ibm01`

<<<<<<< HEAD
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
=======
## 3. Mandatory Codebase Architecture
Cursor, you must implement the complete production-ready MVP. Do NOT generate partial code stubs, placeholder functions, or `# TODO` items. Write clean, highly-optimized, data-oriented Python code using the following file structure:

```text
submissions/fast_mcmc/
├── main.py              # Orchestrator: CLI parsing, process forking, results aggregation
├── worker.py            # Computational Engine: MCMC loop, annealing schedules, Metropolis logic
├── initialization.py    # Constructive Setup: Parallelized GRASP spatial layout routines
├── fast_eval.py         # Mathematics Pipeline: Numba JIT-compiled exact cost delta functions
└── state.py             # Memory Architect: Contiguous NumPy arrays and flat lookups
>>>>>>> attempt2
```

## 4. Deep-Dive Technical Directives

<<<<<<< HEAD
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
=======
### A. Data-Oriented Design (Banning Object Overhead)
- Do NOT create `class Macro` or `class Net`. Traditional object attribute lookups introduce massive Python interpreter overhead.
- All state information must be tracked using flat, contiguous NumPy arrays managed inside `state.py`.
- **Macro Positions:** An $(N, 2)$ float64 array storing the bottom-left $(X, Y)$ coordinates of each macro.
- **Macro Dimensions:** An $(N, 2)$ float64 array storing fixed $(Width, Height)$ dimensions.
- **Network Netlist (CSR Format):** Store the hypergraph using flat integer arrays mimicking Compressed Sparse Row format. Maintain a flat array of net IDs and a corresponding offset pointer array to look up connections in $O(1)$ time.

### B. Strict O(1) Spatial Grid Collision Tracking
- **The Core Constraint:** Pairwise geometric loops (looping through all other macros to verify an intersection) are completely banned during state proposals. 
- **The Matrix Hash Map:** Implement a discrete 2D integer array (`SpatialGrid`) representing the chip canvas divided into uniform rectangular bins. Each cell in the matrix stores the integer `Macro_ID` occupying that spatial zone. Empty cells are marked with a sentinel value of -1.
- **O(1) Collision Evaluation:** When a macro is moved to a candidate coordinate, its bounding box is mapped directly to the matching matrix row and column indices. The engine queries only those exact cells. If any cell contains a valid `Macro_ID` that does not match the moving macro, a collision is flagged instantly.

### C. Parallelized GRASP Initialization
To eliminate sequential bottlenecks on the master process, initialization must happen entirely in parallel on the worker processes.
- **Immediate Forking:** `main.py` parses the layout files and immediately spawns 16 independent worker processes via `multiprocessing.Pool`, passing only raw data, canvas dimensions, and a unique random seed to each.
- **Sequential Greedy Packing:** Inside its isolated memory space, each worker sorts the macro list by area and connectivity degree.
- **Randomized Adaptive Choice:** For each macro, the worker samples a small, bounded set of valid spatial candidate coordinates. It computes a fast bounding box wirelength heuristic for those spots, filters the top $k$ best scoring coordinates, and randomly picks one.
- **Grid Loading:** The selected coordinates are immediately committed to the worker's local `SpatialGrid` matrix. This ensures all 16 workers transition to the annealing phase with highly optimized, yet mathematically distinct, starting layouts.

### D. The Core MCMC Annealing Loop & Metropolis Dynamics
The core execution block inside `worker.py` must run a continuous optimization loop governed by a strict cooling schedule.
- **Biased Perturbation Selection:** Instead of picking macros uniformly, maintain a dynamic weight array. Macros causing high overlap penalties or containing long net connections must be selected for mutations with a higher probability.
- **Mutation Types:** Implement three distinct mutation steps:
  1. *Shift:* Nudge a single macro coordinates by a localized random walk vector.
  2. *Swap:* Select two macros of similar dimensions and exchange their spatial coordinates.
  3. *Reshape:* For soft macros with variable aspect ratios, alter width and height boundaries while keeping total area fixed.
- **Metropolis Criterion:** If a proposed mutation yields a lower proxy cost ($\Delta C < 0$), it is accepted instantly. If the mutation increases the cost ($\Delta C > 0$), it is accepted probabilistically using the exact Boltzmann distribution equation:
  $$P = e^{-\frac{\Delta C}{T}}$$
  A random float $R \in [0, 1)$ is sampled. If $R < P$, the worse layout state is preserved to escape local minima.
- **Geometric Cooling:** The temperature parameter $T$ must decay systematically using the geometric cooling rule:
  $$T_{k+1} = \alpha \times T_k$$
  Set the alpha parameter ($\alpha$) dynamically based on the parsed time limit to ensure the loop exhausts its processing budget efficiently.

### E. High-Precision Proxy Cost Delta Heuristics
Because the full cost function is too slow to evaluate globally, `fast_eval.py` must perform exact localized differential updates compiled with Numba's `@njit(cache=True)`.
- **Exact Wirelength Deltas:** Maintain a flat matrix tracking the bounding box limits `[min_x, max_x, min_y, max_y]` of every net. When a macro shifts, look up its connected net IDs via the flat adjacency list. Recompute the Half-Perimeter Wirelength (HPWL) change *exclusively* for those specific nets.
- **Localized Density & Congestion Deltas:** Query the `SpatialGrid` matrix. Calculate the mathematical difference in cell congestion and density parameters *only* within the localized grid rows and columns that the moving macro exited and entered.

### F. Multi-Tier Overlap Prevention & Final Validity Check
To prevent the algorithm from locking up early while enforcing strict validity at termination, use a tiered rejection structure:
- **Phase 1 (Exploration):** Overlaps are permitted on the spatial grid but are assigned an explicit penalty factor that scales up exponentially as the temperature parameter $T$ drops.
- **Phase 2 (Hard Canvas Filter):** Any mutation that pushes a macro entirely beyond the boundary perimeter of the canvas bounding box is rejected immediately before evaluating any cost math.
- **Phase 3 (Legalization Mode):** When the temperature decays past a critical low threshold, transition the engine to a strict zero-tolerance mode. Any proposed move that introduces a new cell collision in the `SpatialGrid` matrix is dropped instantly.
- **The Final Validity Check:** Before a worker process terminates, it must execute a full, comprehensive verification sweep. It scans its `SpatialGrid` array to ensure total absence of overlapping IDs and checks every macro coordinate against canvas perimeters. If a worker detects a single collision or boundary breach, it must flag its solution as completely invalid, forcing `main.py` to discard it.

### G. Scalable Testing Harness
The orchestrator in `main.py` must expose clear command line interface (CLI) flag overrides using `argparse`:
- `--benchmark (-b)`: String input selecting the target layout problem (e.g., `ibm01`, `ibm02`).
- `--workers (-w)`: Integer parameter capping the maximum size of the multiprocessing pool (allow scaling down to 1 for serial step-through debugging).
- `--timeout (-t)`: Integer parameter specifying execution time limit in seconds (enables rapid 30-second verification runs alongside the full 3600-second tournament runs).

## 5. Implementation Roadmap
Cursor, build out the architecture file by file in this exact sequence. Ensure each file is fully written out with complete logic before proceeding:
1. **`state.py`:** Implement the layout file parser, structural NumPy setups, and the flat CSR netlist lookups.
2. **`fast_eval.py`:** Write the procedural, Numba-compiled (`@njit`) localized HPWL and grid density delta functions.
3. **`initialization.py`:** Implement the fully parallelized GRASP constructive packing routine.
4. **`worker.py`:** Build the complete Metropolis-Hastings loop, biased selector logic, and the final validity checking sweep.
5. **`main.py`:** Code the CLI layer and the `multiprocessing` worker distribution framework.
>>>>>>> attempt2
