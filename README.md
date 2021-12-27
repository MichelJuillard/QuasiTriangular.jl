# QuasiTriangular

A Julia package for quasi upper triangular matrices strongly inspired
by https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/triangular.jl

WORK IN PROGRESS

Type `QuasiUpperTriangular` stores a quasi upper triangular matrix in
a square matrix. The various algorithms ignore the lower zero elements

## Functions

### Matrix - vector product
    - `A_mul_B!(a::QuasiUpperTriangular,b::AbstractVector,work::AbstractVector)`
uses BLAS `trmv` and corrects the results for lower nonzero
elements. For small matrices, (n < 27), uses regular matrix - vector
product
    - `A_mul_B!(A::QuasiUpperTriangular, B::AbstractVector)` pure
    Julia implementation. Doesn't need extra workspace.
### Vector - matrix produc
    - `A_mul_B!(A::AbstractVector, B::QuasiUpperTriangular)` pure
    Julia implementation. Doesn't need extra workspace.

### Transposed matrix -vector product
    - `At_mul_B!(A::QuasiUpperTriangular, B::AbstractVector)` pure
    Julia implementation. Doesn't need extra workspace.

### Vector - transposed matrix product
    - `A_mul_Bt!(A::AbstractVector, B::QuasiUpperTriangular)`  pure
Julia implementation. Doesn't need extra workspace.

### Matrix - matrix product
    - `A_mul_B!(C::AbstractMatrix, A::QuasiUpperTriangular,
      B::AbstractMatrix)`: `C = A*B`, `A` is quasi upper triangular
    - `A_mul_B!(C::AbstractMatrix, A::QuasiUpperTriangular,
	    b::AbstractMatrix)`: `C = αA*B`, `A` is quasi upper triangular
    - `At_mul_B!(C::AbstractMatrix, A::QuasiUpperTriangular,
          B::AbstractMatrix)`: `C = transpose(A)*B`, `A` is quasi upper triangular
    - `At_mul_B!(C::AbstractMatrix, A::QuasiUpperTriangular,
	    b::AbstractMatrix)`: `C = α*transpose(A)*B`, `A` is quasi upper triangular
    - `At_mul_B!(A::QuasiUpperTriangular, B::AbstractMatrix)` `B = transpose(A)*B` in place computation,
    `A` is quasi upper triangular.
    - `A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::QuasiUpperTriangular)`:
    `C = A*B`, `B` is quasi upper triangular.
    - `A_mul_B!(A::AbstractMatrix, B::QuasiUpperTriangular)`:
    `C = A*B`, in place computation, `B` is quasi upper triangular.
    - `A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::QuasiUpperTriangular)`:
    `C = A*transpose(B)`, `B` is quasi upper triangular.
    - `A_mul_Bt!(A::AbstractMatrix, B::QuasiUpperTriangular)`:
    `C = A*transpose(B)`, in place computation, `B` is quasi upper triangular.
    - `A_mul_B!(C:QuasiUpperTriangular, A::QuasiUpperTriangular, B::QuasiUpperTriangular)`,
    `A`, `B` and `C` are quasi upper triangular.

### Linear problem solver
    - `A_ldiv_B!(A::QuasiUpperTriangular, B::AbstractMatrix)` solves `A*X = B`,
    where `A` is quasi upper triangular. Solves by back substitution. Lower off-diagonal elements
    make 2 * 2 problems that are solved explicitely.
    - `A_rdiv_B!(A::AbstractMatrix, B::QuasiUpperTriangular)` solves `X*B = A`,
    where `B` is quasi upper triangular. Solves by back substitution. Lower off-diagonal elements
    make 2 * 2 problems that are solved explicitely.
    - `A_rdiv_Bt!(A::AbstractMatrix, B::QuasiUpperTriangular)` solves `X*transpose(B) = A`,
    where `B` is quasi upper triangular. Solves by back substitution. Lower off-diagonal elements
    make 2 * 2 problems that are solved explicitely.
    - `I_plus_rA_ldiv_B!(r::Float64,a::QuasiUpperTriangular, b::AbstractVector)` solves
    `(I +r*A)*x = b` where `A` is quasi upper triangular.

## TODO
    - replace function names for products by `mul!`
    - introduce lazy transpose evaluation
	- handle quasi lower triangular matrices
	- benchmark cases that have two different implementations
	


