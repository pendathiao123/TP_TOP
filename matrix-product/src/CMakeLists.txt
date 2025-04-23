# Ajouter l'exécutable principal
add_executable(top.matrix_product main.cpp)

# Ajouter les options de compilation
target_sources(top.matrix_product PRIVATE main.cpp)
target_compile_options(top.matrix_product PRIVATE -Wall -Wextra -Wpedantic)

# Ajoutez des options spécifiques au type de build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(top.matrix_product PRIVATE -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(top.matrix_product PRIVATE -O3)
endif()

# Lier les bibliothèques nécessaires
target_link_libraries(top.matrix_product PRIVATE Kokkos::kokkos fmt::fmt)

# Ajouter un exécutable pour les tests
add_executable(test_validation test_validation.cpp)

# Ajouter les options de compilation pour les tests
target_compile_options(test_validation PRIVATE -Wall -Wextra -Wpedantic)

# Lier les bibliothèques nécessaires pour les tests
target_link_libraries(test_validation PRIVATE Kokkos::kokkos fmt::fmt)