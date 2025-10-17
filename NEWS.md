0.2.1
=====
- import Base: Array, Matrix for Julia 1.12

0.2.0
======
- move most functions that operate on submatrices to KroneckerTools package
- rename and refactor all `mul!` functions to be consistent with Julia's post v1.0 conventions
- delete all implementations that use BLAS to keep the core package generic
