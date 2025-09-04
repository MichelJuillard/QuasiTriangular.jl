module QuasiTriangular

using LinearAlgebra
import Base: Array, copy, getindex, Matrix, require_one_based_indexing, setindex!, similar, size
import LinearAlgebra: checksquare, BlasInt, BLAS.@blasfunc, BLAS.libblas

export QuasiUpperTriangular, mul!, ldiv!, rdiv!, I_plus_rA_ldiv_B!, I_plus_rA_plus_sB_ldiv_C!

abstract type AbstractQuasiTriangular{T, S <: AbstractMatrix} <: AbstractMatrix{T} end

struct QuasiUpperTriangular{T, S <: AbstractMatrix{T}} <: AbstractQuasiTriangular{T,S}
    data::S

    function QuasiUpperTriangular{T,S}(data) where {T,S<:AbstractMatrix{T}}
        Base.require_one_based_indexing(data)
        LinearAlgebra.checksquare(data)
        new(data)
    end
end
QuasiUpperTriangular(A::QuasiUpperTriangular) = A
QuasiUpperTriangular{T}(A::QuasiUpperTriangular{T}) where {T} = A
function QuasiUpperTriangular(A::AbstractMatrix)
    return QuasiUpperTriangular{eltype(A), typeof(A)}(A)
end
function QuasiUpperTriangular{T}(A::AbstractMatrix) where T
    QuasiUpperTriangular(convert(AbstractMatrix{T}, A))
end

function QuasiUpperTriangular{T}(A::QuasiUpperTriangular) where T
    Anew = convert(AbstractMatrix{T}, A.data)
    QuasiUpperTriangular(Anew)
end
Matrix(A::QuasiUpperTriangular{T}) where {T} = Matrix{T}(A)

size(A::QuasiUpperTriangular, d) = size(A.data, d)
size(A::QuasiUpperTriangular) = size(A.data)

# For A<:AbstractTriangular, similar(A[, neweltype]) should yield a matrix with the same
# triangular type and underlying storage type as A. The following method covers these cases.
similar(A::QuasiUpperTriangular, ::Type{T}) where {T} = QuasiUpperTriangular(similar(parent(A), T))
# On the other hand, similar(A, [neweltype,] shape...) should yield a matrix of the underlying
# storage type of A (not wrapped in a triangular type). The following method covers these cases.
similar(A::QuasiUpperTriangular, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(A), T, dims)

Array(A::AbstractQuasiTriangular) = Matrix(A)
parent(A:: AbstractQuasiTriangular) = A.data
       
copy(A::QuasiUpperTriangular) = QuasiUpperTriangular(copy(A.data))

real(A::QuasiUpperTriangular{<:Real}) = A
real(A::QuasiUpperTriangular{<:Complex}) = (B = real(A.data); QuasiUpperTriangular(B))

getindex(A::QuasiUpperTriangular{T,S}, i::Integer, j::Integer) where {T,S<:AbstractMatrix{T}} =
    i <= j + 1 ? A.data[i,j] : zero(A.data[j,i])

function setindex!(A::QuasiUpperTriangular, x, i::Integer, j::Integer)
    if i > j + 1
        x == 0 || throw(ArgumentError("cannot set index in the lower triangular part " *
            "($i, $j) of an QuasiUpperTriangular matrix to a nonzero value ($x)"))
    else
        A.data[i,j] = x
    end
    return A
end


## QuasiUpper - Vector multiplication
# Q * v
function mul!(c::AbstractVector, A::QuasiUpperTriangular, b::AbstractVector)
    copy!(c, b)
    m = length(c)
    if m != size(A, 1)
        #TODO correct error msg
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds Ci2 = A.data[1,1]*c[1]
    @inbounds @simd for k = 2:m
        Ci2 += A.data[1,k]*c[k]
    end
    @inbounds for i = 2:m
        Ci1 = A.data[i,i-1]*c[i-1]
        @simd for k = i:m
            Ci1 += A.data[i,k]*c[k]
        end
        c[i-1] = Ci2
        Ci2 = Ci1
    end
    c[m] = Ci2
    return c
end

# Q' * v
function mul!(c::AbstractVector, adjA::Adjoint{T, <:QuasiUpperTriangular}, b::AbstractVector) where T
    A = adjA.parent
    copy!(c, b)
    m, n = size(c, 1), size(c, 2)
    if m != size(A, 1)
        #TODO correct error msg
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[m,m]'c[m,j]
        @simd for k = 1:m - 1
            Bij2 += A.data[k,m]'c[k,j]
        end
        for i = m-1:-1:1
            Bij1 = A.data[i+1,i]'c[i+1,j]
            @simd for k = 1:i
                Bij1 += A.data[k,i]'c[k,j]
            end
            c[i+1,j] = Bij2
            Bij2 = Bij1
        end
        c[1,j] = Bij2
    end
    c
end

## QuasiUpper - Matrix multiplication
# Q * B
function mul!(C::AbstractMatrix, A::QuasiUpperTriangular, B::AbstractMatrix)    
    copy!(C, B)
    m, n = size(C, 1), size(C, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[1,1]*C[1,j]
        @simd for k = 2:m
            Bij2 += A.data[1,k]*C[k,j]
        end
        for i = 2:m
            Bij1 = A.data[i,i-1]*C[i-1,j]
            @simd for k = i:m
                Bij1 += A.data[i,k]*C[k,j]
            end
            C[i-1,j] = Bij2
            Bij2 = Bij1
        end
        C[m,j] = Bij2
    end
    C
end

# Q' * B
function mul!(C::AbstractMatrix, adjA::Adjoint{T, <:QuasiUpperTriangular}, B::AbstractMatrix) where T
    copy!(C, B)
    A = adjA.parent
    m, n = size(C, 1), size(C, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[m,m]'C[m,j]
        @simd for k = 1:m - 1
            Bij2 += A.data[k,m]'C[k,j]
        end
        for i = m-1:-1:1
            Bij1 = A.data[i+1,i]'C[i+1,j]
            @simd for k = 1:i
                Bij1 += A.data[k,i]'C[k,j]
            end
            C[i+1,j] = Bij2
            Bij2 = Bij1
        end
        C[1,j] = Bij2
    end
    C
end

# A * Q
function mul!(C::AbstractMatrix, A::AbstractMatrix, Q::QuasiUpperTriangular)
    C = copy!(C, A)
    m, n = size(C)
    if size(Q, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(Q,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = C[i,n]*Q.data[n,n]
        @simd for k = 1:n - 1
            Aij2 += C[i,k]*Q.data[k,n]
        end
        for j = n-1:-1:1
            Aij1 = C[i,j+1]*Q.data[j+1,j]
            @simd for k = 1:j
                Aij1 += C[i,k]*Q.data[k,j]
            end
            C[i,j+1] = Aij2
            Aij2 = Aij1
        end
        C[i,1] = Aij2
    end
    C
end

# A * Q'
function mul!(C::AbstractMatrix, A::AbstractMatrix, adjB::Adjoint{T, <:QuasiUpperTriangular}) where T
    copy!(C, A)
    B = adjB.parent
    m, n = size(C)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = C[i,1]*B.data[1,1]'
        @simd for k = 2:n
            Aij2 += C[i,k]*B.data[1,k]'
        end
        for j = 2:n
            Aij1 = C[i,j-1]*B.data[j,j-1]'
            @simd for k = j:n
                Aij1 += C[i,k]*B.data[j,k]'
            end
            C[i,j-1] = Aij2
            Aij2 = Aij1
        end
        C[i,n] = Aij2
    end
    C
end


# resolve ambiguities
mul!(C::AbstractMatrix, Q::QuasiUpperTriangular, B::QuasiUpperTriangular) = mul!(C, Q, B.data)
mul!(C::AbstractMatrix, adjA::Adjoint{T, <:QuasiUpperTriangular}, B::QuasiUpperTriangular) where T = mul!(C, adjA, B.data)
mul!(C::AbstractMatrix, A::QuasiUpperTriangular, adjB::Adjoint{T, <:QuasiUpperTriangular}) where T = mul!(C, A.data, adjB)


# solver by substitution
function ldiv!(a::QuasiUpperTriangular, b::AbstractMatrix)
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    @inbounds while j > 0
        if j == 1 || a.data[j,j-1] == 0
            a.data[j,j] == zero(a.data[j,j]) && throw(SingularException(j))
            for k = 1:p
                xj = b[j, k] = a.data[j,j] \ b[j, k]
                @simd for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                    b[i, k] -= a.data[i,j] * xj
                end
            end
            j -= 1
        else
            a11, a21, a12, a22 = a.data[j-1:j,j-1:j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            for k = 1:p
                x2 = m1 * (b[j-1,k] - m2 * b[j,k])
                x1 = b[j-1,k] = a21 \ (b[j,k] - a22 * x2)
                b[j,k] = x2
                @simd for i in j-2:-1:1
                    b[i,k] -= a.data[i,j-1] * x1 + a.data[i,j] * x2
                end
            end
            j -= 2
        end
    end
    b
end

function rdiv!(a::AbstractMatrix, b::QuasiUpperTriangular)
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    @inbounds for i = 1:m
        j = 1
        while  j <= n
            if j == n || b.data[j+1,j] == 0
                b.data[j,j] == zero(b.data[j,j]) && throw(SingularException(j))
                aij = a[i,j]
                @simd for k in 1:j-1
                   aij -= a[i,k] * b.data[k,j]
                end
                a[i,j] = aij/b.data[j,j]
                j += 1
            else
                b11, b21, b12, b22 = b.data[j:j+1,j:j+1]
                det = b11*b22 - b12*b21
                det == zero(b.data[j,j]) && throw(SingularException(j))
                a1 = a[i,j]
                a2 = a[i,j+1]
                @simd for k in 1:j-1
                    a1 -= a[i,k]*b.data[k,j]
                    a2 -= a[i,k]*b.data[k,j+1]
                end
                m1 = -b21/det
                m2 = b22/b21
                a[i,j] = m1 * (a2 - m2 * a1)
                a[i,j+1] = b21 \ (a1 - b11 * a[i,j])
                j += 2
            end
        end
    end
    a
end

function rdiv!(a::AbstractMatrix, adjB::Adjoint{T, <:QuasiUpperTriangular}) where T
    b = adjB.parent
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    @inbounds for i = 1:m
        j = n
        while  j > 0
            if j == 1 || b.data[j,j-1] == 0
                b.data[j,j] == zero(b.data[j,j]) && throw(SingularException(j))
                aij = a[i,j]
                @simd for k = j + 1:n
                    aij -= a[i,k] * b.data[j,k]
                end
                a[i,j] = aij/b.data[j,j]
                j -= 1
            else
                b11, b21, b12, b22 = b.data[j-1:j,j-1:j]
                det = b11*b22 - b12*b21
                det == zero(b.data[j,j]) && throw(SingularException(j))
                a1 = a[i,j-1]
                a2 = a[i,j]
                @simd for k in j + 1:n
                    a1 -= a[i,k]*b.data[j-1,k]
                    a2 -= a[i,k]*b.data[j,k]
                end
                m1 = -b21/det
                m2 = b11/b21
                a[i,j] = m1 * (a1 - m2 * a2)
                a[i,j-1] = b21 \ (a2 - b22 * a[i,j])
                j -= 2
            end
        end
    end
end

# some utility functions for fused operations
function I_plus_rA_ldiv_B!(r::Float64,a::QuasiUpperTriangular, b::AbstractVector)
    m, n = size(a)
    nb, = length(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    @inbounds while j > 0
        if j == 1 || r*a.data[j,j-1] == 0
            pivot = 1.0 + r*a.data[j,j]
            pivot == zero(a.data[j,j]) && throw(SingularException(j))
            b[j] = pivot \ b[j]
            xj = r*b[j]
            @simd for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                b[i] -= a.data[i,j] * xj
            end
            j -= 1
        else
            a11 = 1.0 + r*a.data[j-1,j-1]
            a21 = r*a.data[j,j-1]
            a12 = r*a.data[j-1,j]
            a22 = 1.0 + r*a.data[j,j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            x2 = m1 * (b[j-1] - m2 * b[j])
            x1 = b[j-1] = a21 \ (b[j] - a22 * x2)
            b[j] = x2
            x1 *= r
            x2 *= r
            @simd for i in j-2:-1:1
                b[i] -= a.data[i,j-1] * x1 + a.data[i,j] * x2
            end
            j -= 2
        end
    end
    b
end

function I_plus_rA_plus_sB_ldiv_C!(r::Float64, s::Float64,a::QuasiUpperTriangular, b::QuasiUpperTriangular, c::AbstractVector)
    m, n = size(a)
    nb = length(c)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    @inbounds while j > 0
        if j == 1 || r*a.data[j,j-1] + s*b.data[j,j-1] == 0
            pivot = 1.0 + r*a.data[j,j] + s*b.data[j,j]
            pivot == zero(a.data[j,j]) && throw(SingularException(j))
            c[j] = pivot \ c[j]
            xj1 = r*c[j]
            xj2 = s*c[j]
            @simd for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                c[i] -= a.data[i,j]*xj1 + b.data[i,j]*xj2
            end
            j -= 1
        else
            a11 = 1.0 + r*a.data[j-1,j-1] + s*b.data[j-1,j-1]
            a21 = r*a.data[j,j-1] + s*b.data[j,j-1]
            a12 = r*a.data[j-1,j] + s*b.data[j-1,j]
            a22 = 1.0 + r*a.data[j,j] + s*b.data[j,j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            x2 = m1 * (c[j-1] - m2 * c[j])
            x1 = c[j-1] = a21 \ (c[j] - a22 * x2)
            c[j] = x2
            x11 = r*x1
            x12 = s*x1
            x21 = r*x2
            x22 = s*x2
            @simd for i in j-2:-1:1
                c[i] -= a.data[i,j-1]*x11 + b.data[i,j-1]*x12 + a.data[i,j]*x21 + b.data[i,j]*x22
            end
            j -= 2
        end
    end
end

function I_plus_rA_ldiv_B!(r::Float64,a::QuasiUpperTriangular, b::AbstractMatrix)
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    @inbounds while j > 0
        if j == 1 || r*a.data[j,j-1] == 0
            1.0 + r*a.data[j,j] == zero(a.data[j,j]) && throw(SingularException(j))
            for k = 1:p
                xj = b[j, k] = (1.0 + r*a.data[j,j]) \ b[j, k]
                @simd for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                    b[i, k] -= r*a.data[i,j] * xj
                end
            end
            j -= 1
        else
            a11 = 1.0 + r*a.data[j-1,j-1]
            a21 = r*a.data[j,j-1]
            a12 = r*a.data[j-1,j]
            a22 = 1.0 + r*a.data[j,j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            for k = 1:p
                x2 = m1 * (b[j-1,k] - m2 * b[j,k])
                x1 = b[j-1,k] = a21 \ (b[j,k] - a22 * x2)
                b[j,k] = x2
                @simd for i in j-2:-1:1
                    b[i,k] -= r*(a.data[i,j-1] * x1 + a.data[i,j] * x2)
                end
            end
            j -= 2
        end
    end
end

function I_plus_rA_plus_sB_ldiv_C!(r::Float64, s::Float64,a::QuasiUpperTriangular, b::QuasiUpperTriangular, c::AbstractMatrix)
    m, n = size(a)
    nb, p = size(c)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    @inbounds while j > 0
        if j == 1 || r*a.data[j,j-1] + s*b.data[j,j-1] == 0
            1.0 + r*a.data[j,j] + s*b.data[j,j] == zero(a.data[j,j]) && throw(SingularException(j))
            for k = 1:p
                xj = c[j, k] = (1.0 + r*a.data[j,j] + s*b.data[j,j]) \ c[j, k]
                @simd for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                    c[i, k] -= (r*a.data[i,j] + s*b.data[i,j])* xj
                end
            end
            j -= 1
        else
            a11 = 1.0 + r*a.data[j-1,j-1] + s*b.data[j-1,j-1]
            a21 = r*a.data[j,j-1] + s*b.data[j,j-1]
            a12 = r*a.data[j-1,j] + s*b.data[j-1,j]
            a22 = 1.0 + r*a.data[j,j] + s*b.data[j,j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            for k = 1:p
                x2 = m1 * (c[j-1,k] - m2 * c[j,k])
                x1 = c[j-1,k] = a21 \ (c[j,k] - a22 * x2)
                c[j,k] = x2
                @simd for i in j-2:-1:1
                    c[i,k] -= (r*a.data[i,j-1] + s*b.data[i,j-1]) * x1 + (r*a.data[i,j] + s*b.data[i,j])* x2
                end
            end
            j -= 2
        end
    end
end

end
