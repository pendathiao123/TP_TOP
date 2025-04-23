#ifndef MATRIX_PRODUCT_IMPL_HPP
#define MATRIX_PRODUCT_IMPL_HPP

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>





template <class MatrixType>
auto matrix_init(MatrixType& M) -> void {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  Kokkos::parallel_for(
    "init",
    M.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(M.extent(1)); ++j) {
        M(i, j) = drand48();
      }
    }
  );
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  Kokkos::parallel_for(
    "dgemm_kernel",
    A.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(B.extent(1)); ++j) {
        double acc = 0.0;
        for (int k = 0; k < int(A.extent(1)); ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
        //C(i, j) *= beta + acc;
        C(i, j) = beta * C(i, j) + acc; // Mise à jour correcte

      }
    }
  );
}

//Version avec cache blocking

template <class AMatrixType, class BMatrixType, class CMatrixType>
void matrix_product_blocked(double alpha, const AMatrixType& A, const BMatrixType& B, double beta, CMatrixType& C, int block_size) {
    static_assert(AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "All matrices must be rank 2.");
    
    int M = A.extent(0); // Nombre de lignes de A
    int N = B.extent(1); // Nombre de colonnes de B
    int K = A.extent(1); // Nombre de colonnes de A = nombre de lignes de B

    assert(A.extent(0) == C.extent(0));
    assert(B.extent(1) == C.extent(1));
    assert(A.extent(1) == B.extent(0));

    // Réinitialiser C avec beta
    Kokkos::parallel_for(
        "Initialize C",
        Kokkos::RangePolicy<>(0, M * N),
        KOKKOS_LAMBDA(int idx) {
            int i = idx / N;
            int j = idx % N;
            C(i, j) *= beta;
        });

    // Utilisation de Kokkos::TeamPolicy pour un parallélisme efficace
    Kokkos::parallel_for(
        "Blocked Matrix Multiplication",
        Kokkos::TeamPolicy<>((M + block_size - 1) / block_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int bi = team.league_rank();

            for (int bj = 0; bj < (N + block_size - 1) / block_size; ++bj) {
                for (int bk = 0; bk < (K + block_size - 1) / block_size; ++bk) {
                    // Calcul pour chaque bloc
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, block_size), [&](int i_offset) {
                            int i = bi * block_size + i_offset;
                            if (i >= M) return;

                            for (int j_offset = 0; j_offset < block_size; ++j_offset) {
                                int j = bj * block_size + j_offset;
                                if (j >= N) continue;

                                double acc = 0.0;
                                for (int k_offset = 0; k_offset < block_size; ++k_offset) {
                                    int k = bk * block_size + k_offset;
                                    if (k >= K) continue;

                                    acc += alpha * A(i, k) * B(k, j);
                                }

                                // Mise à jour atomique de C
                                Kokkos::atomic_add(&C(i, j), acc);
                            }
                        });
                }
            }
        });
}
#endif // MATRIX_PRODUCT_IMPL_HPP
