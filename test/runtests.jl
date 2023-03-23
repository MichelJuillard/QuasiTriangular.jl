using LinearAlgebra
using QuasiTriangular
using Random
using Test

Random.seed!(123)

# fills all alternating diagonals after the first subdiagonal with a constant
# to test that garbage values in the lower triangle don't get involved in the computations
function general_quasi(S)
    A = copy(S)
    for n in -2:-2:(-size(A, 1))
        indices = diagind(A, n)
        [setindex!(A, 777, i) for i in indices]
    end
    QuasiUpperTriangular(A)
end

@testset "QuasiTriangular" verbose=true begin
    @testset "QuasiUpper - Matrix multiplication" begin
        n = 7
        S = schur(randn(n, n))
        T = S.T
        Q = general_quasi(T)
        A = randn(n - 1, n)
        B = randn(n, n - 1)
        C1 = similar(B, (n - 1, n))
        C2 = similar(B, (n, n - 1))

        @test mul!(C1, A, Q) ≈ mul!(C1, A, T)
        @test mul!(C1, A, Q') ≈ mul!(C1, A, T')
        @test mul!(C2, Q, B) ≈ mul!(C2, T, B)
        @test mul!(C2, Q', B) ≈ mul!(C2, T', B)

        # assert each method is explicitly defined and we're not just falling back to LinearAlgebra
        @test parentmodule(mul!, typeof.((C1, A, Q))) == QuasiTriangular
        @test parentmodule(mul!, typeof.((C1, A, Q'))) == QuasiTriangular
        @test parentmodule(mul!, typeof.((C2, Q, B))) == QuasiTriangular
        @test parentmodule(mul!, typeof.((C2, Q', B))) == QuasiTriangular
    end

    @testset "QuasiUpper - Vector multiplication" begin
        n = 7
        S = schur(randn(n, n))
        T = S.T
        Q = general_quasi(T)
        a = randn(n)
        b = randn(n)
        c = similar(a)

        @test mul!(c, Q, b) ≈ mul!(c, T, b)
        @test mul!(c, Q', b) ≈ mul!(c, T', b)

        @test parentmodule(mul!, typeof.((c, Q, b))) == QuasiTriangular
        @test parentmodule(mul!, typeof.((c, Q', b))) == QuasiTriangular
    end

    @testset "Extra fused operations" begin
        n = 7
        a = randn(n, n)
        S = schur(a)
        T = S.T
        B = randn(n, n)
        C = similar(B)
        b1 = copy(B)
        x = zeros(n, n)
        ldiv!(general_quasi(T), b1)
        @test T \ B ≈ b1
        b1 = copy(B)
        rdiv!(b1, general_quasi(T))
        @test B / T ≈ b1
        b1 = copy(B)
        rdiv!(b1, general_quasi(T)')
        @test B / T' ≈ b1

        b = rand(n)
        b1 = copy(b)
        r = rand()
        I_plus_rA_ldiv_B!(r, general_quasi(T), b1)
        @test b1 ≈ (Matrix{Float64}(I(n)) + r * T) \ b
        b1 = copy(b)
        s = rand()
        I_plus_rA_plus_sB_ldiv_C!(r, s, general_quasi(T), general_quasi(T * T), b1)
        @test b1 ≈ (Matrix{Float64}(I(n)) + r * T + s * T * T) \ b

        b = rand(n, n)
        b1 = copy(b)
        r = rand()
        I_plus_rA_ldiv_B!(r, general_quasi(T), b1)
        @test b1 ≈ (Matrix{Float64}(I(n)) + r * T) \ b
        b1 = copy(b)
        s = rand()
        I_plus_rA_plus_sB_ldiv_C!(r, s, general_quasi(T), general_quasi(T * T), b1)
        @test b1 ≈ (Matrix{Float64}(I(n)) + r * T + s * T * T) \ b
    end
end
