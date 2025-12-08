# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains hybrid parallel implementations of Monte Carlo π estimation using different parallel computing approaches:

1. **C/C++ Implementation (MPI + OpenMP)**: Traditional HPC approach for CPU clusters
   - **MPI (Message Passing Interface)**: For distributed computing across multiple processes (inter-node parallelism)
   - **OpenMP**: For shared memory parallelism within each process (intra-node parallelism)

2. **Python Implementation (MPI + CUDA)**: Modern GPU-accelerated approach for Google Colab
   - **mpi4py**: Python bindings for MPI (inter-process parallelism)
   - **CUDA (via numba)**: GPU acceleration for massive parallelism

## Build Commands

### C Version: Hybrid MPI+OpenMP
```bash
mpicc -fopenmp -O3 -o monte_carlo_pi_hybrid monte_carlo_pi_hybrid.c -lm
```

Or using the Makefile:
```bash
make
```

Compiler flags:
- `-fopenmp`: Enable OpenMP support
- `-O3`: Optimization level 3
- `-lm`: Link math library

### Python Version: Setup for Google Colab
```bash
# Install dependencies
!apt-get -qq install -y openmpi-bin libopenmpi-dev
!pip install -q mpi4py numba

# Enable GPU in Colab:
# Runtime → Change runtime type → Hardware accelerator → GPU
```

## Execution Commands

### C Version: Running MPI+OpenMP

Basic execution (uses default 10^8 points):
```bash
mpirun -np 4 ./monte_carlo_pi_hybrid
```

With custom number of points:
```bash
mpirun -np 4 ./monte_carlo_pi_hybrid 1000000000
```

Control OpenMP threads per process:
```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./monte_carlo_pi_hybrid 1000000000
```

The above example uses 2 MPI processes × 4 OpenMP threads = 8 total parallel workers.

Using Makefile shortcuts:
```bash
make run           # Run with default settings
make run-large     # Run with 1 billion points
```

### Python Version: Running MPI+CUDA in Colab

Single GPU process (default 10^8 points):
```bash
python monte_carlo_pi_colab.py
```

With custom number of points:
```bash
python monte_carlo_pi_colab.py 1000000000
```

Hybrid MPI+CUDA (multiple processes sharing GPU):
```bash
mpirun -np 2 --allow-run-as-root python monte_carlo_pi_colab.py 1000000000
mpirun -np 4 --allow-run-as-root python monte_carlo_pi_colab.py 1000000000
```

**Note**: In Google Colab, you have access to one GPU. MPI processes share the GPU, distributing work among CPU processes that coordinate GPU computation.

## Code Architecture

### C Implementation: Hybrid MPI+OpenMP ([monte_carlo_pi_hybrid.c](monte_carlo_pi_hybrid.c))

**Two-level parallelism structure:**
1. **MPI level (coarse-grained)**:
   - Work is distributed across MPI processes
   - Each process handles `total_points / num_processes` samples
   - Results are reduced using `MPI_Reduce()` to aggregate counts

2. **OpenMP level (fine-grained)**:
   - Within each MPI process, work is further divided among OpenMP threads
   - Each thread processes `process_points / num_threads` samples
   - Thread-local counts are combined using `reduction(+:local_count)`

**Key implementation details:**
- `MPI_Init_thread()` with `MPI_THREAD_FUNNELED` ensures thread-safe MPI initialization
- Each thread uses `rand_r()` with a unique seed (based on rank + thread_id + time) to avoid correlation
- Timing uses `MPI_Wtime()` for accurate distributed timing with barrier synchronization
- Work distribution handles remainders to ensure all points are processed

**Random number generation:**
- Uses thread-safe `rand_r()` instead of `rand()`
- Seeds incorporate: `time(NULL) + rank * 1000 + thread_id`
- Each thread maintains independent random state (`local_seed`)

### Python Implementation: Hybrid MPI+CUDA ([monte_carlo_pi_colab.py](monte_carlo_pi_colab.py))

**Two-level parallelism structure:**
1. **MPI level (process-level)**:
   - Work distributed across MPI processes using `mpi4py`
   - Each process handles `total_points / num_processes` samples
   - Results aggregated using `comm.reduce()` with `MPI.SUM`

2. **CUDA level (GPU threads)**:
   - Each MPI process launches CUDA kernel on GPU
   - CUDA configuration: 256 threads per block, up to 65535 blocks
   - Each CUDA thread generates and tests multiple points
   - Results collected from GPU memory to host using `copy_to_host()`

**Key implementation details:**
- CUDA kernel (`monte_carlo_kernel`) is decorated with `@cuda.jit` from numba
- Linear Congruential Generator (LCG) used for random numbers in CUDA kernel (GPU-friendly)
- Each CUDA thread has unique seed: `seed_base + thread_idx`
- Remainder points (not divisible by thread count) handled on CPU with NumPy
- GPU availability checked before execution with helpful error messages

**Random number generation in CUDA:**
- Uses LCG with parameters from glibc: `seed = (1103515245 * seed + 12345) & 0x7fffffff`
- Each thread maintains independent seed state
- Generates x and y coordinates separately
- More efficient on GPU than complex RNG algorithms

**Memory management:**
- Device arrays allocated with `cuda.device_array()`
- Results copied back to host with `copy_to_host()`
- Automatic memory cleanup handled by numba

## Expected Output Format

### C Version (MPI+OpenMP)
```
=================================================
Monte Carlo Pi Estimation - Hybrid MPI+OpenMP
=================================================
Total points: 1000000000
MPI processes: 4
OpenMP threads per process: 4
Total parallel workers: 16
=================================================

Results:
-------------------------------------------------
Estimated π:        3.141621982000000
Actual π:           3.141592653589793
Absolute error:     0.000029328410207
Relative error:     0.0009334088%
Points inside:      785405495
Total points:       1000000000
Execution time:     0.234567 seconds
=================================================
```

### Python Version (MPI+CUDA)
```
============================================================
Monte Carlo Pi Estimation - Hybrid MPI+CUDA (Python)
============================================================
Total points: 1,000,000,000
MPI processes: 2
GPU Device: Tesla T4
Compute Capability: (7, 5)
Total Memory: 14.75 GB
============================================================

Results:
------------------------------------------------------------
Estimated π:        3.141567890123456
Actual π:           3.141592653589793
Absolute error:     0.000024763466337
Relative error:     0.0007883421%
Points inside:      785,391,972
Total points:       1,000,000,000
Execution time:     0.456789 seconds
============================================================
```

## Performance Considerations

### General
- **Accuracy**: Use at least 10^7 points for reasonable accuracy; 10^8-10^9 for better estimates
- **Load balancing**: Work is evenly distributed; remainders are handled by giving extra points to first few processes/threads

### C Version (MPI+OpenMP)
- **Scaling**:
  - MPI scales across nodes (distributed memory)
  - OpenMP scales within nodes (shared memory)
  - Optimal configuration depends on hardware topology
- **Best for**: Traditional HPC clusters, CPU-based systems
- **Typical speedup**: Near-linear with number of cores (up to hardware limits)

### Python Version (MPI+CUDA)
- **Scaling**:
  - GPU provides massive parallelism (thousands of CUDA threads)
  - MPI can distribute work across multiple GPUs (if available)
  - In Colab: Single GPU shared by MPI processes
- **Best for**: GPU-accelerated systems, cloud environments like Google Colab
- **Typical speedup**: 10-100x vs single CPU core, depending on GPU
- **Memory**: Limited by GPU memory (typically 12-16 GB in Colab)

## Google Colab Usage

### Quick Start with Jupyter Notebook
1. Upload [colab_setup.ipynb](colab_setup.ipynb) to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run cells sequentially to install dependencies and execute

### Direct Python Script Usage
1. Upload [monte_carlo_pi_colab.py](monte_carlo_pi_colab.py)
2. Install dependencies in a cell:
   ```python
   !apt-get -qq install -y openmpi-bin libopenmpi-dev
   !pip install -q mpi4py numba
   ```
3. Run: `!python monte_carlo_pi_colab.py 1000000000`

### Important Colab Notes
- GPU access is limited to T4 (free tier) or better (Colab Pro)
- Runtime may disconnect after inactivity
- Only one GPU available per session
- MPI processes share the same GPU

## File Overview

- [monte_carlo_pi_hybrid.c](monte_carlo_pi_hybrid.c) - C implementation with MPI+OpenMP
- [monte_carlo_pi_colab.py](monte_carlo_pi_colab.py) - Python implementation with MPI+CUDA for Colab
- [colab_setup.ipynb](colab_setup.ipynb) - Jupyter notebook with step-by-step Colab setup
- [Makefile](Makefile) - Build automation for C version
- [CLAUDE.md](CLAUDE.md) - This file

## Development Notes

### C Version
- Accepts one command-line argument: number of points (optional, defaults to 10^8)
- Error handling includes validation for positive point counts and MPI initialization checks
- All parallel workers (MPI processes × OpenMP threads) are reported in output for transparency

### Python Version
- Works with or without MPI (degrades gracefully to single-process mode)
- Requires GPU for CUDA acceleration (provides helpful error if not available)
- Accepts one command-line argument: number of points (optional, defaults to 10^8)
- Prints GPU information (model, compute capability, memory) when available
