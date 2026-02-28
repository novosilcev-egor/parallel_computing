#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

class Matrix {
private:
    std::vector<int> data;
    size_t rows;
    size_t cols;

    void fill_out() {
        for (size_t i = 0; i < rows * cols; ++i) {
            // Генерируем целое число в диапазоне [1, 100]
            data[i] = rand() % 100 + 1;
        }
    }

public:
    Matrix(size_t r, size_t c, bool fill = true) : rows(r), cols(c), data(r * c, 0) {
        if (fill) fill_out();
    }

    int* raw() { return data.data(); }

    static void local_multiply(const int* A_block, size_t local_rows,
                               const int* B, size_t N, size_t K,
                               int* C_block) {
        std::memset(C_block, 0, local_rows * K * sizeof(int));

        const size_t BLOCK_SIZE = 64;

        for (size_t kk = 0; kk < N; kk += BLOCK_SIZE) {
            size_t k_end = std::min(kk + BLOCK_SIZE, N);
            for (size_t ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
                size_t i_end = std::min(ii + BLOCK_SIZE, local_rows);
                for (size_t jj = 0; jj < K; jj += BLOCK_SIZE) {
                    size_t j_end = std::min(jj + BLOCK_SIZE, K);
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t k = kk; k < k_end; ++k) {
                            int a_ik = A_block[i * N + k];
                            for (size_t j = jj; j < j_end; ++j) {
                                C_block[i * K + j] += a_ik * B[k * K + j];
                            }
                        }
                    }
                }
            }
        }
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t M = 1000; // строки первой матрицы
    const size_t N = 1000; // столбцы первой матрицы / строки второй матрицы
    const size_t K = 1000; // столбцы второй матрицы
    const int NUM_EXPERIMENTS = 10;

    if (rank == 0) {
        std::cout << "Тестирование умножения матриц " << M << "x" << N
                  << " на " << N << "x" << K << std::endl;
        std::cout << "Количество MPI-процессов: " << size << std::endl;
        std::cout << "Количество экспериментов: " << NUM_EXPERIMENTS << std::endl << std::endl;
    }

    std::vector<int> sendcounts(size), displs(size);
    std::vector<int> recvcounts(size), rdispls(size);

    size_t base_rows = M / size;
    size_t remainder = M % size;

    for (int i = 0; i < size; ++i) {
        size_t rows_i = base_rows + (static_cast<size_t>(i) < remainder ? 1 : 0);
        sendcounts[i] = static_cast<int>(rows_i * N);
        recvcounts[i] = static_cast<int>(rows_i * K);
    }
    displs[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

    size_t local_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    std::vector<int> A_data, C_data;
    std::vector<int> B_data(N * K);
    std::vector<int> A_local(local_rows * N);
    std::vector<int> C_local(local_rows * K);

    double total_time = 0.0;

    for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
        if (rank == 0) {
            Matrix A(M, N);
            Matrix B(N, K);
            A_data.assign(A.raw(), A.raw() + M * N);
            std::memcpy(B_data.data(), B.raw(), N * K * sizeof(int));
            C_data.resize(M * K);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Bcast(B_data.data(), static_cast<int>(N * K), MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Scatterv(rank == 0 ? A_data.data() : nullptr,
                     sendcounts.data(), displs.data(), MPI_INT,
                     A_local.data(), static_cast<int>(local_rows * N), MPI_INT,
                     0, MPI_COMM_WORLD);

        Matrix::local_multiply(A_local.data(), local_rows,
                               B_data.data(), N, K,
                               C_local.data());

        MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * K), MPI_INT,
                    rank == 0 ? C_data.data() : nullptr,
                    recvcounts.data(), rdispls.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            total_time += elapsed;
            std::cout << "Эксперимент " << std::setw(2) << exp + 1 << ": "
                      << std::fixed << std::setprecision(3) << elapsed << " секунд" << std::endl;
        }
    }

    if (rank == 0) {
        double average_time = total_time / NUM_EXPERIMENTS;
        std::cout << "\nСреднее время выполнения: " << std::fixed << std::setprecision(3)
                  << average_time << " секунд" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
