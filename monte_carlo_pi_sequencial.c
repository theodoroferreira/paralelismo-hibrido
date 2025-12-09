#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.141592653589793238462643

/**
 * Função que gera pontos aleatórios e conta quantos estão dentro do círculo unitário
 *
 * @param num_points Número de pontos a serem gerados
 * @param seed Semente para o gerador de números aleatórios
 * @return Número de pontos dentro do círculo
 */
long long monte_carlo_count(long long num_points, unsigned int seed) {
    long long count_inside = 0;

    // Usando uma semente única para o gerador de números aleatórios
    unsigned int local_seed = seed;

    for (long long i = 0; i < num_points; i++) {
        // Gerar um ponto aleatório em [0,1] × [0,1]
        double x = (double)rand_r(&local_seed) / RAND_MAX;
        double y = (double)rand_r(&local_seed) / RAND_MAX;

        // Verificar se o ponto está dentro do círculo unitário
        if (x * x + y * y <= 1.0) {
            count_inside++;
        }
    }

    return count_inside;
}

int main(int argc, char *argv[]) {
    long long total_points = 100000000; // Default: 10^8 pontos
    long long points_per_process;
    long long local_count = 0;
    long long global_count = 0;
    double pi_estimate, error;
    double start_time, end_time;

    // Parse argumentos da linha de comando
    if (argc > 1) {
        total_points = atoll(argv[1]);
        if (total_points <= 0) {
            fprintf(stderr, "Erro: O número de pontos deve ser positivo\n");
            return 1;
        }
    }

    // Gerar pontos e contar quantos estão dentro do círculo
    start_time = clock();
    unsigned int seed = (unsigned int)time(NULL); // Usando o tempo atual como semente para o gerador de números aleatórios

    global_count = monte_carlo_count(total_points, seed);

    end_time = clock();

    // Calcular e exibir os resultados
    pi_estimate = 4.0 * (double)global_count / (double)total_points;
    error = fabs(pi_estimate - PI);
    double error_percentage = (error / PI) * 100.0;

    printf("Resultados:\n");
    printf("-------------------------------------------------\n");
    printf("Estimativa de π:      %.15f\n", pi_estimate);
    printf("π real:               %.15f\n", PI);
    printf("Erro absoluto:        %.15f\n", error);
    printf("Erro relativo:        %.10f%%\n", error_percentage);
    printf("Pontos dentro:        %lld\n", global_count);
    printf("Pontos totais:        %lld\n", total_points);
    printf("Tempo de execução:    %.6f segundos\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    printf("=================================================\n");

    return 0;
}
