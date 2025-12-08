MPICC = mpicc

CFLAGS = -fopenmp -O3 -Wall -Wextra
LDFLAGS = -lm

TARGET = monte_carlo_pi_hybrid

SRC = monte_carlo_pi_hybrid.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo ""
	@echo "Build successful!"
	@echo "Run with: mpirun -np <processes> ./$(TARGET) [num_points]"
	@echo "Example: mpirun -np 4 ./$(TARGET) 1000000000"
	@echo ""

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@echo "Running with 4 MPI processes, 4 OpenMP threads per process..."
	OMP_NUM_THREADS=4 mpirun -np 4 ./$(TARGET)

run-large: $(TARGET)
	@echo "Running with 1 billion points..."
	OMP_NUM_THREADS=4 mpirun -np 4 ./$(TARGET) 1000000000

help:
	@echo "Monte Carlo Pi Estimation - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make            - Build the hybrid MPI+OpenMP version"
	@echo "  make clean      - Remove built executables"
	@echo "  make run        - Build and run with default settings"
	@echo "  make run-large  - Build and run with 1 billion points"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Manual execution examples:"
	@echo "  mpirun -np 2 ./$(TARGET)"
	@echo "  OMP_NUM_THREADS=8 mpirun -np 4 ./$(TARGET) 500000000"
	@echo ""

.PHONY: all clean run run-large help
