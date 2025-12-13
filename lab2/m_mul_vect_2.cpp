#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include "/opt/homebrew/opt/libomp/include/omp.h"

class Matrix {
private:
    std::vector<std::vector<int>> data;
    size_t rows;
    size_t cols;

    void fill_out() {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // Генерируем целое число в диапазоне [1, 100]
                data[i][j] = rand() % 100 + 1;
            }
        }
    }

    // Оптимизированное блочное умножение матриц
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("Размерности матриц не подходят для умножения");
        }

        Matrix result(rows, other.cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                result.data[i][j] = 0;
            }
        }

        const size_t BLOCK_SIZE = 64; // оптимизировано под L1-кэш

        for (size_t kk = 0; kk < cols; kk += BLOCK_SIZE) {
            size_t k_end = std::min(kk + BLOCK_SIZE, cols);
            #pragma omp parallel for collapse(4) schedule(guided)
            for (size_t ii = 0; ii < rows; ii += BLOCK_SIZE) {
                for (size_t jj = 0; jj < other.cols; jj += BLOCK_SIZE) {
                    size_t i_end = std::min(ii + BLOCK_SIZE, rows);
                    size_t j_end = std::min(jj + BLOCK_SIZE, other.cols);
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t j = jj; j < j_end; ++j) {
                            int sum = 0;
                            #pragma omp simd reduction(+:sum)
                            for (size_t k = kk; k < k_end; ++k) {
                                sum += data[i][k] * other.data[k][j];
                            }
                            result.data[i][j] += sum;
                        }
                    }
                }
            }
        }

        return result;
    }

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<int>(cols));
        fill_out();
    }

    Matrix operator*(const Matrix& other) const {
        return multiply(other);
    }

};

int main() {
    const size_t M = 1000; // строки первой матрицы
    const size_t N = 1000; // столбцы первой матрицы / строки второй матрицы
    const size_t K = 1000; // столбцы второй матрицы
    const int NUM_EXPERIMENTS = 10;
    
    std::cout << "Тестирование умножения матриц " << M << "x" << N << " на " << N << "x" << K << std::endl;
    std::cout << "Количество экспериментов: " << NUM_EXPERIMENTS << std::endl << std::endl;
    
    double total_time = 0.0;
    
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        Matrix A(M, N);
        Matrix B(N, K);
        
        double start = omp_get_wtime();
        Matrix C = A * B;
        double end = omp_get_wtime();
        
        double time = end - start;
        total_time += time;
        
        std::cout << "Эксперимент " << std::setw(2) << i + 1 << ": " 
                  << std::fixed << std::setprecision(3) << time << " секунд" << std::endl;
    }
    
    double average_time = total_time / NUM_EXPERIMENTS;
    std::cout << "\nСреднее время выполнения: " << std::fixed << std::setprecision(3) 
              << average_time << " секунд" << std::endl;

    return 0;
}
