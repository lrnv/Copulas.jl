"""
    BetaCopula{d, MT}

Fields:
- `ranks::MT` - ranks matrix (d × n), each row contains integers 1..n

Constructor

    BetaCopula(u)
    
The empirical beta copula in dimension ``d`` is defined as

```math
C_n^{\\beta}(u) = \\frac{1}{n} \\sum_{i=1}^n \\prod_{j=1}^d F_{n,R_{ij}}(u_j),
```

where ``R_{ij}`` is the rank of observation ``i`` in margin ``j``, and ``U \\sim Beta(r, n+1-r)``.

Notes:
- This is always a valid copula for any finite sample size `n`.
- Supports `cdf`, `logpdf` at observed points and random sampling.

References:
* [segers2017empirical](@cite) Segers, J., Sibuya, M., & Tsukahara, H. (2017). The empirical beta copula. Journal of Multivariate Analysis, 155, 35-51.
"""
struct BetaCopula{d,MT} <: Copula{d}
    ranks::MT   # d×n (each row is in 1..n)
    n::Int
    function BetaCopula(data::AbstractMatrix)
        U = _as_pxn(size(data,1), data)      # d×n, pseudo-observations
        R = _rowwise_ordinalranks(U)         # d×n, integers 1..n (if no ties)
        # Quick check: each row should be a permutation of 1..n
        @inbounds for j in 1:size(R,1)
            if !all(sort(@view R[j,:]) .== 1:size(R,2))
                @warn "Row $j of ranks is not a permutation of 1..n; marginals may not be uniform."
            end
        end
        return new{size(U,1), typeof(R)}(R, size(U,2))
    end
end

# Row-wise ordinal ranks 1..n per variable
function _rowwise_ordinalranks(U::AbstractMatrix)
    d, n = size(U)
    R = Matrix{Int}(undef, d, n)
    @inbounds for j in 1:d
        R[j, :] = StatsBase.ordinalrank(@view U[j, :])
    end
    return R
end


# =========================================
#  Basis via stable recurrences (AD- and boundary-safe)
# =========================================
# p_{n,k}(u) = C(n,k) u^k (1-u)^{n-k}, k=0..n
# Bernstein basis degree n — avoids 0^0 and uses stable recurrences.

function _bernvec_n(u::T, n::Int) where {T<:Real}
    v = Vector{T}(undef, n+1)
    if u == zero(T)
        v[1] = one(T); @inbounds for k in 2:n+1 v[k]=zero(T) end
        return v
    elseif u == one(T)
        @inbounds for k in 1:n v[k]=zero(T) end; v[end]=one(T)
        return v
    end
    one_minus_u = one(T) - u
    p = one_minus_u^n
    v[1] = p
    @inbounds for k in 0:n-1
        p *= ((n - k) / (k + 1)) * (u / one_minus_u)
        v[k+2] = p
    end
    return v
end

# Beta PDF with integer parameters: f(u; r, n+1-r) = n * p_{n-1, r-1}(u)
function _beta_pdf_basis(u::T, n::Int) where {T<:Real}
    v = Vector{T}(undef, n)  # índices r=1..n
    if u == zero(T)
        v[1] = T(n); @inbounds for r in 2:n v[r]=zero(T) end
        return v
    elseif u == one(T)
        @inbounds for r in 1:n-1 v[r]=zero(T) end; v[n]=T(n)
        return v
    end
    p = _bernvec_n(u, n-1)               # k=0..n-1
    @inbounds for r in 1:n
        v[r] = T(n) * p[r]               # r ↔ k=r-1
    end
    return v
end

# Beta CDF with integer parameters: F_{n,r}(u) = ∑_{s=r}^{n} p_{n,s}(u)
function _beta_cdf_basis(u::T, n::Int) where {T<:Real}
    p = _bernvec_n(u, n)                 # p[k+1] ↔ p_{n,k}(u), k=0..n
    tail = Vector{T}(undef, n+1)
    acc = zero(T)
    @inbounds for k in n:-1:0
        acc += p[k+1]
        tail[k+1] = acc                   # tail[k+1] = ∑_{s=k}^n p_{n,s}(u)
    end
    v = Vector{T}(undef, n)              # r = 1..n → tail[r+1] = ∑_{s=r}^n p_{n,s}(u)
    @inbounds for r in 1:n
        v[r] = tail[r+1]                 # <-- índice corregido
    end
    return v
end

# ========
#   CDF
# ========

function Distributions.cdf(C::BetaCopula{d}, u::AbstractVector) where {d}
    @assert length(u) == d
    n = C.n
    # tablas por dimensión en el punto u
    CDFtab = ntuple(j -> _beta_cdf_basis(eltype(u)(u[j]), n), d)
    total = zero(eltype(first(CDFtab)))
    @inbounds for i in 1:n
        prod_term = one(total)
        @inbounds for j in 1:d
            r = C.ranks[j,i]     # 1..n
            prod_term *= CDFtab[j][r]
            if prod_term == 0
                break
            end
        end
        total += prod_term
    end
    return total / n
end

# =========
#  LOGPDF
# =========

function Distributions.logpdf(C::BetaCopula{d}, u::AbstractVector) where {d}
    @assert length(u) == d
    n = C.n
    PDFtab = ntuple(j -> _beta_pdf_basis(eltype(u)(u[j]), n), d)
    dens = zero(eltype(first(PDFtab)))
    @inbounds for i in 1:n
        prod_term = one(dens)
        @inbounds for j in 1:d
            r = C.ranks[j,i]
            prod_term *= PDFtab[j][r]
            if prod_term == 0
                break
            end
        end
        dens += prod_term
    end
    dens /= n
    dens = max(dens, zero(dens))
    return log(dens + eps(eltype(dens)))
end

# =========
#  RAND!
# =========

function Distributions._rand!(rng::Distributions.AbstractRNG,
                              C::BetaCopula{d},
                              u::AbstractVector{T}) where {d,T<:Real}
    i = Distributions.rand(rng, 1:C.n)  # choose a mixture component
    @inbounds for j in 1:d
        r = C.ranks[j,i]
        u[j] = Distributions.rand(rng, Distributions.Beta(r, C.n + 1 - r))
    end
    return u
end

# ===============================
#  Conditioning fast path (distortion)
# ===============================

@inline function DistortionFromCop(C::BetaCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,MT,p}
    # Build conditional mixture weights over observations given U_js = u_js.
    # w_i ∝ ∏_{t=1..p} BetaPDF(u_jt; r_{j_t,i}, n+1−r_{j_t,i})
    @assert 1 <= i <= D
    n = C.n
    w = zeros(Float64, n)
    @inbounds for idx in 1:n
        prodw = 1.0
        @inbounds for (t, j) in pairs(js)
            r = C.ranks[j, idx]
            prodw *= Distributions.pdf(Distributions.Beta(r, n + 1 - r), uⱼₛ[t])
        end
        w[idx] = prodw
    end
    s = sum(w)
    if s <= 0
        return NoDistortion()
    end
    w ./= s
    comps = Vector{Distributions.Beta}(undef, n)
    @inbounds for idx in 1:n
        r = C.ranks[i, idx]
        comps[idx] = Distributions.Beta(r, n + 1 - r)
    end
    return Distributions.MixtureModel(comps, w)
end
