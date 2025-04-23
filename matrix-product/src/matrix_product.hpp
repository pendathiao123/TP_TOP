#ifndef MATRIX_PRODUCT_HPP
#define MATRIX_PRODUCT_HPP

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cstdlib>
#include <cmath>

// Typedefs for matrix views
using MatrixA = Kokkos::View<double**, Kokkos::LayoutLeft>;
using MatrixB = Kokkos::View<double**, Kokkos::LayoutRight>;
using MatrixC = Kokkos::View<double**, Kokkos::LayoutLeft>;

// Function to initialize a matrix template
template <class MatrixType>
void matrix_init(MatrixType& M);

// Basic matrix product function
template <class AMatrixType, class BMatrixType, class CMatrixType>
void matrix_product(double alpha, const AMatrixType& A, const BMatrixType& B, double beta, CMatrixType& C);

// Blocked matrix product function
template <class AMatrixType, class BMatrixType, class CMatrixType>
void matrix_product_blocked(double alpha, const AMatrixType& A, const BMatrixType& B, double beta, CMatrixType& C, int block_size);

#include "matrix_product_impl.hpp"

#endif // MATRIX_PRODUCT_HPP
