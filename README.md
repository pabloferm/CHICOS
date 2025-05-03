# CHICOS (Caley-Hamilton and Invariants for OScillations)

To compile the C++ version and requires the EIGEN package.

`
g++ -std=c++17 -O3 -march=native -ffast-math -I /usr/include/eigen3 CHICOS.cpp -o chicos
`

`
g++ -std=c++17 -O3 -Wall -Wextra -lm -fopenmp -I /usr/include/eigen3 CHICOS.cpp -o chicos
`
