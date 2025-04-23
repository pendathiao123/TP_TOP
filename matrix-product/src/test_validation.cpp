#include "matrix_product.hpp"
#include <Kokkos_Core.hpp>
#include <fmt/core.h>
#include <cmath>
#include <iostream>

// Fonction pour valider le produit matriciel
bool validate_matrix_product(int m, int n, int k, double alpha, double beta) {
    auto A = MatrixA("A", m, k);
    auto B = MatrixB("B", k, n);
    auto C_blocked = MatrixC("C_blocked", m, n);
    auto C_reference = MatrixC("C_reference", m, n);

    // Initialisation des matrices
    matrix_init(A);
    matrix_init(B);
    Kokkos::deep_copy(C_blocked, 0.0);
    Kokkos::deep_copy(C_reference, 0.0);

    // Calcul des produits matriciels
    matrix_product_blocked(alpha, A, B, beta, C_blocked, 64); // Méthode optimisée
    matrix_product(alpha, A, B, beta, C_reference);           // Méthode de référence

    // Validation des résultats
    bool correct = true;
    Kokkos::parallel_reduce(
        "Verify Results",
        m,
        KOKKOS_LAMBDA(int i, bool& local_correct) {
            for (int j = 0; j < n; ++j) {
                double diff = std::abs(C_blocked(i, j) - C_reference(i, j));
                double tol = 1e-6 * std::abs(C_reference(i, j)); // Tolérance relative
                if (diff > 1e-6 && diff > tol) {
                    local_correct = false;
                }
            }
        },
        Kokkos::LAnd<bool>(correct)
    );

    // Imprime les différences si la validation échoue
    if (!correct) {
        Kokkos::parallel_for(
            "Print Differences",
            m,
            KOKKOS_LAMBDA(int i) {
                for (int j = 0; j < n; ++j) {
                    double diff = std::abs(C_blocked(i, j) - C_reference(i, j));
                    double tol = 1e-6 * std::abs(C_reference(i, j));
                    if (diff > 1e-6 && diff > tol) {
                        printf("Mismatch at (%d, %d): C_blocked = %f, C_reference = %f, diff = %f\n",
                               i, j, C_blocked(i, j), C_reference(i, j), diff);
                    }
                }
            }
        );
    }

    return correct;
}

// Fonction principale
auto main(int argc, char* argv[]) -> int {
    Kokkos::initialize(argc, argv);
    {
        // Exécute la validation avec des tailles spécifiques
        if (!validate_matrix_product(100, 100, 100, 1.0, 0.0)) {
            fmt::print("Validation failed.\n");
            return -1;
        }
    }
    Kokkos::finalize();
    fmt::print("Validation passed.\n");
    return 0;
}