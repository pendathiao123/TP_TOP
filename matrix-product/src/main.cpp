#include "matrix_product.hpp"
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <fmt/core.h>

using MatrixA = Kokkos::View<double**, Kokkos::LayoutLeft>;
using MatrixB = Kokkos::View<double**, Kokkos::LayoutRight>;
using MatrixC = Kokkos::View<double**, Kokkos::LayoutLeft>;


auto main(int argc, char* argv[]) -> int {
    if (argc < 4) {
        fmt::print("Usage: {} <M> <N> <K>\n", argv[0]);
        return -1;
    }
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    srand48(42); // Seed for deterministic RNG

    Kokkos::initialize(argc, argv);
    {
        auto A = MatrixA("A", m, k);
        auto B = MatrixB("B", k, n);
        auto C = MatrixC("C", m, n);

        double alpha = drand48();
        matrix_init(A);
        matrix_init(B);
        double beta = drand48();
        matrix_init(C);

        Kokkos::fence();

        // Configurations à tester
        std::vector<int> block_sizes = {32, 64, 128, 256};
        std::vector<int> thread_counts = {1, 2, 3, 4}; // Nombre de threads à tester

        // Sauvegarder les résultats dans un fichier CSV
        std::ofstream results_file("results_threads_blocks.csv");
        results_file << "BlockSize,Threads,Time,Performance\n";

        for (int block_size : block_sizes) {
            for (int num_threads : thread_counts) {
                setenv("OMP_NUM_THREADS", std::to_string(num_threads).c_str(), 1);

                fmt::print("Testing with block size: {} and threads: {}\n", block_size, num_threads);

                auto start = std::chrono::high_resolution_clock::now();
                matrix_product_blocked(alpha, A, B, beta, C, block_size);
                Kokkos::fence();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double time = elapsed.count();

                double flops = 2.0 * m * n * k;
                double performance = (flops / time) / 1e9;

                fmt::print("Blocked matrix product completed in {:.6f} seconds.\n", time);
                fmt::print("Performance: {:.2f} GFLOP/s\n", performance);

                // Écrire les résultats dans le fichier
                results_file << block_size << "," << num_threads << "," << time << "," << performance << "\n";
            }
        }

        results_file.close();
        fmt::print("Results saved to results_threads_blocks.csv\n");
    }
    Kokkos::finalize();
    return 0;
}
    
// Teste avec une seule taille de bloque 
/*
    auto main(int argc, char* argv[]) -> int {
      if (argc < 4) {
          fmt::print("Usage: {} <M> <N> <K>\n", argv[0]);
          return -1;
      }
      int m = std::atoi(argv[1]);
      int n = std::atoi(argv[2]);
      int k = std::atoi(argv[3]);
  
      srand48(42); // Seed for deterministic RNG
  
      Kokkos::initialize(argc, argv);
      {
          auto A = MatrixA("A", m, k);
          auto B = MatrixB("B", k, n);
          auto C = MatrixC("C", m, n);
  
          double alpha = drand48();
          matrix_init(A);
          matrix_init(B);
          double beta = drand48();
          matrix_init(C);
  
          Kokkos::fence();
  
          // Taille de bloc et nombre de threads à tester
          int block_size = 64; // Taille de bloc fixe
          int num_threads = 4; // Nombre de threads fixe
  
          setenv("OMP_NUM_THREADS", std::to_string(num_threads).c_str(), 1);
  
          fmt::print("Testing with block size: {} and threads: {}\n", block_size, num_threads);
  
          auto start = std::chrono::high_resolution_clock::now();
          matrix_product_blocked(alpha, A, B, beta, C, block_size);
          Kokkos::fence();
          auto end = std::chrono::high_resolution_clock::now();
  
          std::chrono::duration<double> elapsed = end - start;
          double time = elapsed.count();
  
          double flops = 2.0 * m * n * k;
          double performance = (flops / time) / 1e9;
  
          fmt::print("Blocked matrix product completed in {:.6f} seconds.\n", time);
          fmt::print("Performance: {:.2f} GFLOP/s\n", performance);
      }
      Kokkos::finalize();
      return 0;
  }*/