module QuasiTriangular
## QuasiUpperTriangular Matrix of real numbers

using LinearAlgebra
import Base.size
import Base.similar
import Base.getindex
import Base.require_one_based_indexing
import Base.setindex!
import Base.copy
import Base.Matrix
import Base.strides
import Base.elsize
import Base.unsafe_convert
import LinearAlgebra.checksquare
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.BLAS.libblas

export QuasiUpperTriangular, I_plus_rA_ldiv_B!, I_plus_rA_plus_sB_ldiv_C!, A_ldiv_B!, A_rdiv_B!, A_rdiv_Bt!, A_mul_B!, A_mul_Bt!, At_mul_B!

abstract type AbstractQuasiTriangular{T, S <: AbstractMatrix} <: AbstractMatrix{T} end

struct QuasiUpperTriangular{T, S <: AbstractMatrix{T}} <: AbstractQuasiTriangular{T,S}
    data::S

    function QuasiUpperTriangular{T,S}(data) where {T,S<:AbstractMatrix{T}}
        require_one_based_indexing(data)
        checksquare(data)
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
            "($i, $j) of an UpperTriangular matrix to a nonzero value ($x)"))
    else
        A.data[i,j] = x
    end
    return A
end

## Implement StridedArray interface to be compatible with generic BLAS calls
Base.strides(a::QuasiUpperTriangular{T, <: AbstractMatrix{T}}) where {T} = strides(a.data)
Base.elsize(::Type{<:QuasiUpperTriangular{T, <: AbstractMatrix{T}}})  where T = Base.elsize(Matrix{T})
Base.unsafe_convert(::Type{Ptr{T}}, a::QuasiUpperTriangular{T, <: AbstractMatrix{T}}) where {T} = pointer(a.data)

## Generic quasi triangular right vector, multiplication
function A_mul_B!(a::QuasiUpperTriangular,b::AbstractVector,work::AbstractVector)
    if size(a,1) < 27
        # pure Julia is faster
        A_mul_B!(a,b)
        return
    end
    copy!(work,b)
    BLAS.trmv!('U','N','N',a.data,b)
    @inbounds @simd for i= 2:size(a,1)
        b[i] += a[i,i-1]*work[i-1]
    end
end

# pure Julia implementation right multiplication
function A_mul_B!(A::QuasiUpperTriangular, B::AbstractVector)
    m = length(B)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds Bi2 = A.data[1,1]*B[1]
    @inbounds @simd for k = 2:m
        Bi2 += A.data[1,k]*B[k]
    end
    @inbounds for i = 2:m
        Bi1 = A.data[i,i-1]*B[i-1]
        @simd for k = i:m
            Bi1 += A.data[i,k]*B[k]
        end
        B[i-1] = Bi2
        Bi2 = Bi1
    end
    B[m] = Bi2
end

# pure Julia transpose implementation right vector multiplication
function At_mul_B!(A::QuasiUpperTriangular, B::AbstractVector)
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[m,m]'B[m,j]
        @simd for k = 1:m - 1
            Bij2 += A.data[k,m]'B[k,j]
        end
        for i = m-1:-1:1
            Bij1 = A.data[i+1,i]'B[i+1,j]
            @simd for k = 1:i
                Bij1 += A.data[k,i]'B[k,j]
            end
            B[i+1,j] = Bij2
            Bij2 = Bij1
        end
        B[1,j] = Bij2
    end
    B
end

# Pure Julia implementation left vector multiplication
function A_mul_B!(A::AbstractVector, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = A[i,n]*B.data[n,n]
        @simd for k = 1:n - 1
            Aij2 += A[i,k]*B.data[k,n]
        end
        for j = n-1:-1:1
            Aij1 = A[i,j+1]*B.data[j+1,j]
            @simd for k = 1:j
                Aij1 += A[i,k]*B.data[k,j]
            end
            A[i,j+1] = Aij2
            Aij2 = Aij1
        end
        A[i,1] = Aij2
    end
    A
end

# Pure Julia implementation transpose left vector multiplication
function A_mul_Bt!(A::AbstractVector, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = A[i,1]*B.data[1,1]'
        @simd for k = 2:n
            Aij2 += A[i,k]*B.data[1,k]'
        end
        for j = 2:n
            Aij1 = A[i,j-1]*B.data[j,j-1]'
            @simd for k = j:n
                Aij1 += A[i,k]*B.data[j,k]'
            end
            A[i,j-1] = Aij2
            Aij2 = Aij1
        end
        A[i,n] = Aij2
    end
    A
end

# right matrix multiplication c = a*b
function A_mul_B!(c::AbstractMatrix, a::QuasiUpperTriangular, b::AbstractMatrix)
    A_mul_B!(c, 1.0, a, b)
    c
end

# right matrix multiplication on view in VecOrMat
function A_mul_B!(c::AbstractVecOrMat, a::QuasiUpperTriangular, b::AbstractVecOrMat, nr::Int64, nc::Int64)
    A_mul_B!(c, 1.0, a, b, nr, nc)
    c
end

# c = alpha*a*b
function A_mul_B!(c::AbstractMatrix, alpha::Float64, a::QuasiUpperTriangular, b::AbstractMatrix)
    m, n = size(b)
    if size(a, 1) != m
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(b,1))"))
    end
    copy!(c,b)
    BLAS.trmm!('L','U','N','N',alpha,a.data,c)

    @inbounds for i= 2:m
        x = alpha*a[i,i-1]
        @simd for j=1:n
            c[i,j] += x*b[i-1,j]
        end
    end
    c
end

# c = alpha*a*b in view for VecOrMat
function A_mul_B!(c::AbstractVecOrMat, alpha::Float64, a::QuasiUpperTriangular, b::AbstractVecOrMat, nr::Int64, nc::Int64)
    m, n = size(a)
    copy!(c,b)
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('L'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(nr), Ref{BlasInt}(nc),
          Ref{Float64}(alpha), a.data, Ref{BlasInt}(nr), c, Ref{BlasInt}(nr))

    b1 = reshape(b,nr,nc)
    c1 = reshape(c,nr,nc)
    @inbounds for i= 2:m
        x = a[i,i-1]
        @simd for j=1:m
            c1[i,j] += x*b1[i-1,j]
        end
    end
    c
end

# right matrix multiplication
function A_mul_B!(A::QuasiUpperTriangular, B::AbstractMatrix)
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[1,1]*B[1,j]
        @simd for k = 2:m
            Bij2 += A.data[1,k]*B[k,j]
        end
        for i = 2:m
            Bij1 = A.data[i,i-1]*B[i-1,j]
            @simd for k = i:m
                Bij1 += A.data[i,k]*B[k,j]
            end
            B[i-1,j] = Bij2
            Bij2 = Bij1
        end
        B[m,j] = Bij2
    end
    B
end

function At_mul_B!(c::AbstractMatrix, a::QuasiUpperTriangular, b::AbstractMatrix)
    At_mul_B!(c,1.0,a,b)
    c
end

function At_mul_B!(c::AbstractMatrix, alpha::Float64, a::QuasiUpperTriangular, b::AbstractMatrix)
    m, n = size(b)
    if size(a, 1) != m
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(b,1))"))
    end
    copy!(c,b)
    BLAS.trmm!('L','U','T','N',alpha,a.data,c)

    @inbounds for i= 1:m-1
        x = alpha*a[i+1,i]
        @simd for j=1:n
            c[i,j] += x*b[i+1,j]
        end
    end
    c
end

function At_mul_B!(A::QuasiUpperTriangular, B::AbstractMatrix)
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        Bij2 = A.data[m,m]'B[m,j]
        @simd for k = 1:m - 1
            Bij2 += A.data[k,m]'B[k,j]
        end
        for i = m-1:-1:1
            Bij1 = A.data[i+1,i]'B[i+1,j]
            @simd for k = 1:i
                Bij1 += A.data[k,i]'B[k,j]
            end
            B[i+1,j] = Bij2
            Bij2 = Bij1
        end
        B[1,j] = Bij2
    end
    B
end

function A_mul_B!(c::AbstractMatrix, a::AbstractMatrix, b::QuasiUpperTriangular)
    A_mul_B!(c,1.0,a,b)
    c
end

function A_mul_B!(c::AbstractVecOrMat, a::AbstractVecOrMat, b::QuasiUpperTriangular, nr::Int64, nc::Int64)
    A_mul_B!(c,1.0,a,b,nr,nc)
    c
end

function A_mul_B!(c::AbstractMatrix, alpha::Float64, a::AbstractMatrix, b::QuasiUpperTriangular)
    m, n = size(a)
    if size(b, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    copy!(c,a)
    BLAS.trmm!('R','U','N','N',alpha,b.data,c)

    @inbounds for i= 2:n
        x = alpha*b[i,i-1]
        @simd for j=1:m
            c[j,i-1] += x*a[j,i]
        end
    end
    c
end

function A_mul_B!(c::AbstractVecOrMat, alpha::Float64, a::AbstractVecOrMat, b::QuasiUpperTriangular, nr::Int64, nc::Int64)
    m, n = size(b)
    copy!(c,a)
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('R'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(nr), Ref{BlasInt}(nc),
          Ref{Float64}(alpha), b.data, Ref{BlasInt}(nc), c, Ref{BlasInt}(nr))

    a1 = reshape(a,nr,nc)
    c1 = reshape(c,nr,nc)
    @inbounds for i= 2:m
        x = b[i,i-1]
        @simd for j=1:m
            c1[j,i-1] += x*a1[j,i]
        end
    end
    c
end

function A_mul_B!(A::AbstractMatrix, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = A[i,n]*B.data[n,n]
        @simd for k = 1:n - 1
            Aij2 += A[i,k]*B.data[k,n]
        end
        for j = n-1:-1:1
            Aij1 = A[i,j+1]*B.data[j+1,j]
            @simd for k = 1:j
                Aij1 += A[i,k]*B.data[k,j]
            end
            A[i,j+1] = Aij2
            Aij2 = Aij1
        end
        A[i,1] = Aij2
    end
    A
end

function A_mul_Bt!(c::AbstractMatrix, a::AbstractMatrix, b::QuasiUpperTriangular)
    A_mul_Bt!(c,1.0,a,b)
    c
end

function A_mul_Bt!(c::AbstractMatrix, alpha::Float64, a::AbstractMatrix, b::QuasiUpperTriangular)
    m, n = size(a)
    if size(b, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    copy!(c,a)
    BLAS.trmm!('R','U','T','N',alpha,b.data,c)
    @inbounds for j= 2:n
        x = alpha*b[j,j-1]
        @simd for i=1:m
            c[i,j] += x*a[i,j-1]
        end
    end
    c
end

function A_mul_Bt!(A::AbstractMatrix, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    @inbounds for i = 1:m
        Aij2 = A[i,1]*B.data[1,1]'
        @simd for k = 2:n
            Aij2 += A[i,k]*B.data[1,k]'
        end
        for j = 2:n
            Aij1 = A[i,j-1]*B.data[j,j-1]'
            @simd for k = j:n
                Aij1 += A[i,k]*B.data[j,k]'
            end
            A[i,j-1] = Aij2
            Aij2 = Aij1
        end
        A[i,n] = Aij2
    end
    A
end

A_mul_B!(c::QuasiUpperTriangular,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = A_mul_B!(c.data,a,b.data)
At_mul_B!(c::QuasiUpperTriangular,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = At_mul_B!(c.data,a,b.data)
A_mul_Bt!(c::QuasiUpperTriangular,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = A_mul_Bt!(c.data,a,b.data)
A_mul_B!(c::AbstractMatrix,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = A_mul_B!(c,a,b.data)
At_mul_B!(c::AbstractMatrix,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = At_mul_B!(c,a,b.data)
A_mul_Bt!(c::AbstractMatrix,a::QuasiUpperTriangular,b::QuasiUpperTriangular) = A_mul_Bt!(c,a,b.data)

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64, a::AbstractVecOrMat{Float64}, offset_a::Int64,
                  ma::Int64, na::Int64, b::QuasiUpperTriangular, offset_b::Int64, nb::Int64)
    m, n = size(b)
    copyto!(c, offset_c, a, offset_a, ma*na)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('R'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(ma), Ref{BlasInt}(na),
          Ref{Float64}(alpha), b.data, Ref{BlasInt}(na), Ref(c, offset_c), Ref{BlasInt}(ma))

    inda = offset_a + ma
    indc = offset_c
    @inbounds for i = 2:na
        x = b[i,i-1]
        @simd for j = 1:ma
            c[indc] += x*a[inda]
            inda += 1
            indc += 1
        end
    end
end

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64, a::AbstractVecOrMat{Float64}, offset_a::Int64,
                  ma::Int64, na::Int64, b::QuasiUpperTriangular)
    nb = size(b, 2)
    A_mul_B!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64, a::QuasiUpperTriangular, offset_a::Int64,
                  ma::Int64, na::Int64, b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    copyto!(c, offset_c, b, offset_b, na*nb)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('L'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(ma), Ref{BlasInt}(nb),
          Ref{Float64}(alpha), a.data, Ref{BlasInt}(ma), Ref(c, offset_c), Ref{BlasInt}(na))

    @inbounds for i = 2:ma
        x = a[i,i-1]
        indb = offset_b
        indc = offset_c + 1
        @simd for j = 1:nb
            c[indc] += x*b[indb]
            indb += ma
            indc += ma
        end
    end
end

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64, a::QuasiUpperTriangular,
                  b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ma, na = size(a)
    A_mul_B!(c, offset_c, a, 1, ma, na, b, offset_b, nb)
end

#=
function A_mul_B!(c::Array{Float64,1}, offset_c::Int64,
                  a::QuasiUpperTriangular, offset_a::Int64, ma::Int64, na::Int64,
                  b::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}, offset_b::Int64, nb::Int64)
    copyto!(c, offset_c, b, offset_b, na*nb)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('L'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(na), Ref{BlasInt}(nb),
          Ref{Float64}(alpha), a.data, Ref{BlasInt}(ma), Ref(c, offset_c), Ref{BlasInt}(na))

    @inbounds for i= 2:ma
        x = a[i,i-1]
        indb = offset_b
        indc = offset_c + 1
        @simd for j=1:nb
            c[indc] += x*b[indb]
            indb += ma
            indc += ma
        end
        offset_b +=  1
        offset_c += 1
    end
    c
end


function A_mul_B!(c::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}, offset_c::Int64,
                  a::QuasiUpperTriangular, offset_a::Int64, ma::Int64, na::Int64,
                  b::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}, offset_b::Int64, nb::Int64)
    copyto!(c, offset_c, b, offset_b, na*nb)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('L'), Ref{UInt8}('U'), Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{BlasInt}(na), Ref{BlasInt}(nb),
          Ref{Float64}(alpha), a.data, Ref{BlasInt}(ma), Ref(c, offset_c), Ref{BlasInt}(na))

    offsetb_orig = offset_b
    offsetc_orig = offset_c
    @inbounds for i= 2:ma
        x = a[i,i-1]
        indb = offset_b
        indc = offset_c +1
        @simd for j=1:nb
            c[indc] += x*b[indb]
            indb += ma
            indc += ma
        end
        offset_b += 1
        offset_c += 1
    end
    c
end
=#

function At_mul_B!(c::VecOrMat{Float64}, offset_c::Int64, a::QuasiUpperTriangular{Float64}, offset_a::Int64,
                  ma::Int64, na::Int64, b::VecOrMat{Float64}, offset_b::Int64, nb::Int64)
    copyto!(c, offset_c, b, offset_b, ma*nb)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          Ref{UInt8}('L'), Ref{UInt8}('U'), Ref{UInt8}('T'), Ref{UInt8}('N'), Ref{BlasInt}(ma), Ref{BlasInt}(nb),
          Ref{Float64}(alpha), a.data, Ref{BlasInt}(ma), Ref(c, offset_c), Ref{BlasInt}(ma))

    @inbounds for i = 2:ma
        x = a[i,i-1]
        indb = offset_b + i - 1
        indc = offset_c + i - 2
        @simd for j = 1:nb
            c[indc] += x*b[indb]
            indb += ma
            indc += ma
        end
    end
end

function A_mul_Bt!(c::AbstractVector{Float64}, offset_c::Int64, a::AbstractVector{Float64}, offset_a::Int64, ma::Int64, na::Int64, b::QuasiUpperTriangular{Float64}, offset_b::Int64, nb::Int64)
    copyto!(c, offset_c, a, offset_a, ma*na)
    alpha = 1.0
    ccall((@blasfunc(dtrmm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}),
          'R', 'U', 'T', 'N', ma, nb,
          alpha, b.data, nb, Ref(c, offset_c), ma)

    inda = offset_a
    indc = offset_c + ma
    @inbounds for j= 2:na
        x = alpha*b[j,j-1]
        @simd for i=1:ma
            c[indc] += x*a[inda]
            inda += 1
            indc += 1
        end
    end
    c
end

function A_mul_Bt!(c::AbstractVector{Float64}, offset_c::Int64, a::AbstractVector{Float64}, offset_a::Int64, ma::Int64, na::Int64, b::QuasiUpperTriangular{Float64})
    nb = size(b, 1)
    A_mul_Bt!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

# solver by substitution
function A_ldiv_B!(a::QuasiUpperTriangular, b::AbstractMatrix)
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
end


function A_rdiv_B!(a::AbstractMatrix, b::QuasiUpperTriangular)
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
end

function A_rdiv_Bt!(a::AbstractMatrix, b::QuasiUpperTriangular)
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
