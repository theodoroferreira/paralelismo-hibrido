# Estimativa de π por Monte Carlo - Computação Paralela Híbrida

Implementação híbrida de estimativa de π usando Monte Carlo com computação paralela.

## Implementação

### C - MPI + OpenMP
Abordagem tradicional de HPC para clusters de CPU:
- **MPI**: Paralelismo distribuído entre processos
- **OpenMP**: Paralelismo de memória compartilhada dentro de cada processo

## Compilação e Execução

### C (MPI+OpenMP)

**Compilar:**
```bash
make
# ou manualmente:
mpicc -fopenmp -O3 -o monte_carlo_pi_hybrid monte_carlo_pi_hybrid.c -lm
```

**Executar:**
```bash
# Execução básica (10^8 pontos)
mpirun -np 4 ./monte_carlo_pi_hybrid

# Com número customizado de pontos
mpirun -np 4 ./monte_carlo_pi_hybrid 1000000000

# Controlar threads OpenMP por processo
export OMP_NUM_THREADS=4
mpirun -np 2 ./monte_carlo_pi_hybrid 1000000000
# Exemplo acima: 2 processos MPI × 4 threads OpenMP = 8 workers paralelos

# Usando Makefile
make run           # Execução padrão
make run-large     # Execução com 1 bilhão de pontos
```

## Arquitetura do Código

### C - Paralelismo em dois níveis:
1. **Nível MPI** (granularidade grossa): Distribui trabalho entre processos
2. **Nível OpenMP** (granularidade fina): Divide trabalho entre threads dentro de cada processo

## Exemplo de Saída

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

## Considerações de Performance

- **Precisão**: Use pelo menos 10^7 pontos para precisão razoável; 10^8-10^9 para melhores estimativas
- **Versão C**: Melhor para clusters HPC tradicionais baseados em CPU

## Arquivos do Projeto

- [monte_carlo_pi_hybrid.c](monte_carlo_pi_hybrid.c) - Implementação C com MPI+OpenMP
- [Makefile](Makefile) - Automação de build para C

## Requisitos

### Versão C:
- MPI (OpenMPI ou MPICH)
- Compilador C com suporte a OpenMP (gcc/clang)
