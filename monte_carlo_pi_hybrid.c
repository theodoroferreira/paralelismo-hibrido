/*
 * Monte Carlo Pi Estimation - Hybrid MPI + OpenMP Implementation
 *
 * This program estimates the value of π using the Monte Carlo method
 * with hybrid parallelism combining MPI (inter-node) and OpenMP (intra-node).
 *
 * Method:
 * 1. Generate random points in the unit square [0,1] × [0,1]
 * 2. Count points falling inside the unit circle (x² + y² ≤ 1)
 * 3. Estimate π ≈ 4 × (points_inside / total_points)
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.141592653589793238462643

/**
 * Generate random points and count those inside the unit circle
 *
 * @param num_points Number of points to generate in this thread
 * @param seed Random seed (unique per thread)
 * @return Number of points inside the circle
 */
long long monte_carlo_count(long long num_points, unsigned int seed) {
    long long count_inside = 0;

    // Each thread uses its own random state
    unsigned int local_seed = seed;

    for (long long i = 0; i < num_points; i++) {
        // Generate random point in [0,1] × [0,1]
        double x = (double)rand_r(&local_seed) / RAND_MAX;
        double y = (double)rand_r(&local_seed) / RAND_MAX;

        // Check if point is inside unit circle
        if (x * x + y * y <= 1.0) {
            count_inside++;
        }
    }

    return count_inside;
}

int main(int argc, char *argv[]) {
    int rank, size;
    long long total_points = 100000000; // Default: 10^8 points
    long long points_per_process;
    long long local_count = 0;
    long long global_count = 0;
    double pi_estimate, error;
    double start_time, end_time;
    int num_threads;

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    if (argc > 1) {
        total_points = atoll(argv[1]);
        if (total_points <= 0) {
            if (rank == 0) {
                fprintf(stderr, "Error: Number of points must be positive\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Set number of OpenMP threads (can be controlled by OMP_NUM_THREADS environment variable)
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }

    // Distribute work among MPI processes
    points_per_process = total_points / size;
    long long remainder = total_points % size;

    // First 'remainder' processes get one extra point
    if (rank < remainder) {
        points_per_process++;
    }

    if (rank == 0) {
        printf("=================================================\n");
        printf("Monte Carlo Pi Estimation - Hybrid MPI+OpenMP\n");
        printf("=================================================\n");
        printf("Total points: %lld\n", total_points);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Total parallel workers: %d\n", size * num_threads);
        printf("=================================================\n\n");
    }

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Parallel Monte Carlo computation using OpenMP within each MPI process
    #pragma omp parallel reduction(+:local_count)
    {
        int thread_id = omp_get_thread_num();
        int threads = omp_get_num_threads();

        // Distribute points among threads
        long long points_per_thread = points_per_process / threads;
        long long thread_remainder = points_per_process % threads;

        if (thread_id < thread_remainder) {
            points_per_thread++;
        }

        // Create unique seed for this thread based on rank and thread_id
        unsigned int seed = (unsigned int)(time(NULL) + rank * 1000 + thread_id);

        // Count points inside circle for this thread
        long long thread_count = monte_carlo_count(points_per_thread, seed);
        local_count += thread_count;
    }

    // Reduce results from all MPI processes to rank 0
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Calculate and display results on rank 0
    if (rank == 0) {
        pi_estimate = 4.0 * (double)global_count / (double)total_points;
        error = fabs(pi_estimate - PI);
        double error_percentage = (error / PI) * 100.0;

        printf("Results:\n");
        printf("-------------------------------------------------\n");
        printf("Estimated π:        %.15f\n", pi_estimate);
        printf("Actual π:           %.15f\n", PI);
        printf("Absolute error:     %.15f\n", error);
        printf("Relative error:     %.10f%%\n", error_percentage);
        printf("Points inside:      %lld\n", global_count);
        printf("Total points:       %lld\n", total_points);
        printf("Execution time:     %.6f seconds\n", end_time - start_time);
        printf("=================================================\n");
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
