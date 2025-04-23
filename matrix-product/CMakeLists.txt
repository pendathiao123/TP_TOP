cmake_minimum_required(VERSION 3.25)
project(top.matrix_product LANGUAGES CXX)

# Définit le type de build par défaut si non spécifié
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()

# Affiche le type de build sélectionné
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Trouver OpenMP
find_package(OpenMP REQUIRED)

# Inclure les dépendances externes
include(FetchContent)
FetchContent_Declare(Kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos GIT_TAG 4.6.00)
FetchContent_MakeAvailable(Kokkos)
FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt GIT_TAG 11.1.4)
FetchContent_MakeAvailable(fmt)

# Ajouter le sous-répertoire src
add_subdirectory(src)

# Activer les tests
enable_testing()

# Ajouter un test pour vérifier les résultats
add_test(NAME VerifyResults COMMAND test_validation)