MPICC = mpicc
CC = gcc

CFLAGS = -fopenmp -O3 -Wall -Wextra
LDFLAGS = -lm

TARGET_HYBRID = monte_carlo_pi_hybrid
TARGET_SEQUENCIAL = monte_carlo_pi_sequencial

SRC_HYBRID = monte_carlo_pi_hybrid.c
SRC_SEQUENCIAL = monte_carlo_pi_sequencial.c

# Alvo padrão
all: $(TARGET_HYBRID) $(TARGET_SEQUENCIAL)

# Compilação da versão híbrida (MPI + OpenMP)
$(TARGET_HYBRID): $(SRC_HYBRID)
	$(MPICC) $(CFLAGS) -o $(TARGET_HYBRID) $(SRC_HYBRID) $(LDFLAGS)
	@echo ""
	@echo "Build successful for hybrid version!"
	@echo "Run with: mpirun -np <processes> ./$(TARGET_HYBRID) [num_points]"
	@echo "Example: mpirun -np 4 ./$(TARGET_HYBRID) 1000000000"
	@echo ""

# Compilação da versão sequencial
$(TARGET_SEQUENCIAL): $(SRC_SEQUENCIAL)
	$(CC) $(CFLAGS) -o $(TARGET_SEQUENCIAL) $(SRC_SEQUENCIAL) $(LDFLAGS)
	@echo ""
	@echo "Build successful for sequential version!"
	@echo "Run with: ./$(TARGET_SEQUENCIAL) [num_points]"
	@echo "Example: ./$(TARGET_SEQUENCIAL) 1000000000"
	@echo ""

# Limpeza dos arquivos compilados
clean:
	rm -f $(TARGET_HYBRID) $(TARGET_SEQUENCIAL)

# Executar a versão híbrida com configurações padrão
run: $(TARGET_HYBRID)
	@echo "Running hybrid version with 4 MPI processes, 4 OpenMP threads per process..."
	OMP_NUM_THREADS=4 mpirun -np 4 ./$(TARGET_HYBRID)

# Executar a versão híbrida com 1 bilhão de pontos
run-large: $(TARGET_HYBRID)
	@echo "Running hybrid version with 1 billion points..."
	OMP_NUM_THREADS=4 mpirun -np 4 ./$(TARGET_HYBRID) 1000000000

# Executar a versão sequencial com configurações padrão
run-sequencial: $(TARGET_SEQUENCIAL)
	@echo "Running sequential version..."
	./$(TARGET_SEQUENCIAL)

# Executar a versão sequencial com 1 bilhão de pontos
run-sequencial-large: $(TARGET_SEQUENCIAL)
	@echo "Running sequential version with 1 billion points..."
	./$(TARGET_SEQUENCIAL) 1000000000

# Mostrar a ajuda
help:
	@echo "Monte Carlo Pi Estimation - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make            - Build both hybrid (MPI + OpenMP) and sequential versions"
	@echo "  make clean      - Remove built executables"
	@echo "  make run        - Build and run hybrid version with default settings"
	@echo "  make run-large  - Build and run hybrid version with 1 billion points"
	@echo "  make run-sequencial - Build and run sequential version with default settings"
	@echo "  make run-sequencial-large - Build and run sequential version with 1 billion points"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Manual execution examples:"
	@echo "  mpirun -np 2 ./$(TARGET_HYBRID)"
	@echo "  OMP_NUM_THREADS=8 mpirun -np 4 ./$(TARGET_HYBRID) 500000000"
	@echo "  ./$(TARGET_SEQUENCIAL)"
	@echo "  ./$(TARGET_SEQUENCIAL) 500000000"
	@echo ""

.PHONY: all clean run run-large run-sequencial run-sequencial-large help
