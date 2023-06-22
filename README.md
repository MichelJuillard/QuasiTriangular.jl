# QuasiTriangular

A Julia package for quasi upper triangular matrices strongly inspired
by https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/triangular.jl

WORK IN PROGRESS

Type `QuasiUpperTriangular` stores a quasi upper triangular matrix in
a square matrix. The various algorithms ignore the lower zero elements

## Functions
Conventional `mul!` functions are defined to allow normal multiplication using `*`

lets define $Q$ as a QuasiUpperTriangular matrix.

### Matrix - vector product
- $Q * \vec{v}$
- $Q^T * \vec{v}$

### Matrix - Matrix product
- $Q * {A}$
- $Q^T * A$
- $A * Q$
- $A * Q^T$


### Linear problem solvers
- `ldiv!(Q, A)` solves $Q*X = A$,
    Solves by back substitution. Lower off-diagonal elements make 2 * 2 problems that are solved explicitly.
- `rdiv!(A, Q)` solves $X*Q = A$,
    Solves by back substitution. Lower off-diagonal elements make 2 * 2 problems that are solved explicitly.
- `rdiv!(A, Q')` solves $X*Q^T = A$,
    Solves by back substitution. Lower off-diagonal elements make 2 * 2 problems that are solved explicitly.

lets define $r$ and $s$ as floats
     
> Note: these functions break conventions and mutate their last argument
- `I_plus_rA_ldiv_B!(r, Q, b)` solves $(I + rQ)*\vec{x} = \vec{b}$
- `I_plus_rA_ldiv_B!(r, Q, B)` solves $(I + rQ)*X = B$ 
- `I_plus_rA_plus_sB_ldiv_C!(r, s, Q1, Q2, c)` solves $(I + rQ_1 + sQ_2)*\vec{x} = \vec{c}$
- `I_plus_rA_plus_sB_ldiv_C!(r, s, Q1, Q2, C)` solves $(I + rQ_1 + sQ_2)*X = C$

## TODO
- [ ] assert that sub-diagonal does not contain consecutive non-zero elements 
- [ ] handle quasi lower triangular matrices
- [ ] profile, benchmark, and reintroduce BLAS based implementations if needed (for specific strided-matrix element-types)
