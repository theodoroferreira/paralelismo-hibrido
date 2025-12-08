"""
Monte Carlo Pi Estimation - Hybrid MPI + CUDA Implementation for Google Colab

This program estimates the value of π using the Monte Carlo method
with hybrid parallelism combining mpi4py (inter-node) and CUDA (GPU acceleration).

Method:
1. Generate random points in the unit square [0,1] × [0,1]
2. Count points falling inside the unit circle (x² + y² ≤ 1)
3. Estimate π ≈ 4 × (points_inside / total_points)

Requirements:
- mpi4py: MPI for Python
- numba: CUDA JIT compilation
- numpy: Numerical operations

Installation in Colab:
!apt-get -qq install -y openmpi-bin libopenmpi-dev
!pip install -q mpi4py numba

Note: Google Colab provides GPU support. Enable it via:
Runtime → Change runtime type → Hardware accelerator → GPU
"""

import numpy as np
from numba import cuda
import math
import time
import sys

# Try to import MPI, but make it optional for single-process runs
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in single-process mode")


# CUDA kernel for Monte Carlo simulation
@cuda.jit
def monte_carlo_kernel(points_inside, total_points_per_thread, seed_base):
    """
    CUDA kernel to generate random points and count those inside the unit circle.

    Each thread generates random points and counts how many fall inside the circle.
    Uses a simple LCG (Linear Congruential Generator) for random numbers.

    Args:
        points_inside: Output array to store count of points inside circle per thread
        total_points_per_thread: Number of points each thread should generate
        seed_base: Base seed for random number generation
    """
    # Get thread index
    idx = cuda.grid(1)

    if idx < points_inside.size:
        # Initialize random seed for this thread
        # LCG parameters (same as glibc)
        seed = seed_base + idx
        count = 0

        # Generate and test points
        for i in range(total_points_per_thread):
            # Linear Congruential Generator for random numbers
            # x_n+1 = (a * x_n + c) mod m
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            x = (seed & 0xFFFF) / 65536.0  # Random x in [0,1]

            seed = (1103515245 * seed + 12345) & 0x7fffffff
            y = (seed & 0xFFFF) / 65536.0  # Random y in [0,1]

            # Check if point is inside unit circle
            if x * x + y * y <= 1.0:
                count += 1

        # Store result
        points_inside[idx] = count


def monte_carlo_cuda(total_points, rank=0, size=1):
    """
    Run Monte Carlo simulation on GPU using CUDA.

    Args:
        total_points: Total number of points to generate
        rank: MPI rank (process ID)
        size: Total number of MPI processes

    Returns:
        Number of points inside the circle for this process
    """
    # Check if CUDA is available
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Please enable GPU in Colab: Runtime → Change runtime type → GPU")

    # Distribute work among MPI processes
    points_per_process = total_points // size
    remainder = total_points % size

    # First 'remainder' processes get one extra point
    if rank < remainder:
        points_per_process += 1

    # CUDA configuration
    threads_per_block = 256
    blocks = min((points_per_process + threads_per_block - 1) // threads_per_block, 65535)
    total_threads = threads_per_block * blocks

    # Distribute points among CUDA threads
    points_per_thread = points_per_process // total_threads
    thread_remainder = points_per_process % total_threads

    # Allocate device memory
    d_points_inside = cuda.device_array(total_threads, dtype=np.int64)

    # Generate unique seed for this process (ensure it fits in 32-bit range)
    seed_base = (int(time.time() * 1000) + rank * 100000) & 0x7FFFFFFF

    # Launch CUDA kernel
    monte_carlo_kernel[blocks, threads_per_block](d_points_inside, points_per_thread, seed_base)

    # Copy results back to host
    h_points_inside = d_points_inside.copy_to_host()

    # Handle remainder points (run on CPU to avoid complexity)
    cpu_points = thread_remainder
    if cpu_points > 0:
        # Ensure seed is within NumPy's acceptable range [0, 2^32 - 1]
        cpu_seed = (seed_base + total_threads) % (2**32)
        np.random.seed(cpu_seed)
        x = np.random.random(cpu_points)
        y = np.random.random(cpu_points)
        cpu_count = np.sum(x*x + y*y <= 1.0)
    else:
        cpu_count = 0

    # Sum all counts
    local_count = np.sum(h_points_inside) + cpu_count

    return local_count


def print_gpu_info():
    """Print information about available GPU."""
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"GPU Device: {device.name.decode()}")
        print(f"Compute Capability: {device.compute_capability}")
        print(f"Total Memory: {device.total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available")


def main():
    """Main function to run hybrid MPI+CUDA Monte Carlo simulation."""

    # Initialize MPI if available
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
        comm = None

    # Default number of points
    total_points = 100000000  # 10^8

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            total_points = int(sys.argv[1])
            if total_points <= 0:
                if rank == 0:
                    print("Error: Number of points must be positive", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            if rank == 0:
                print("Error: Invalid number format", file=sys.stderr)
            sys.exit(1)

    # Print header (only rank 0)
    if rank == 0:
        print("=" * 60)
        print("Monte Carlo Pi Estimation - Hybrid MPI+CUDA (Python)")
        print("=" * 60)
        print(f"Total points: {total_points:,}")
        print(f"MPI processes: {size}")

        if cuda.is_available():
            print_gpu_info()

        print("=" * 60)
        print()

    # Synchronize all processes before timing
    if MPI_AVAILABLE:
        comm.Barrier()

    # Start timing
    start_time = time.time()

    # Run Monte Carlo simulation on GPU
    try:
        local_count = monte_carlo_cuda(total_points, rank, size)
    except RuntimeError as e:
        if rank == 0:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Reduce results from all MPI processes
    if MPI_AVAILABLE:
        global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    else:
        global_count = local_count

    # Synchronize and stop timing
    if MPI_AVAILABLE:
        comm.Barrier()
    end_time = time.time()

    # Calculate and display results (only rank 0)
    if rank == 0:
        pi_estimate = 4.0 * global_count / total_points
        pi_actual = math.pi
        error = abs(pi_estimate - pi_actual)
        error_percentage = (error / pi_actual) * 100.0
        execution_time = end_time - start_time

        print("Results:")
        print("-" * 60)
        print(f"Estimated π:        {pi_estimate:.15f}")
        print(f"Actual π:           {pi_actual:.15f}")
        print(f"Absolute error:     {error:.15f}")
        print(f"Relative error:     {error_percentage:.10f}%")
        print(f"Points inside:      {global_count:,}")
        print(f"Total points:       {total_points:,}")
        print(f"Execution time:     {execution_time:.6f} seconds")
        print("=" * 60)


if __name__ == "__main__":
    main()
