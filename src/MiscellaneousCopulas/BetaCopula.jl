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
        d, n = size(data)
        R = Matrix{Int}(undef, d, n)
        @inbounds for j in 1:d
            R[j, :] = StatsBase.ordinalrank(@view data[j, :])
        end
        return new{d, typeof(R)}(R, n)
    end
end

# =========================================
#  Basis via stable recurrences (AD- and boundary-safe)
# =========================================
# p_{n,k}(u) = C(n,k) u^k (1-u)^{n-k}, k=0..n
# Bernstein basis degree n — avoids 0^0 and uses stable recurrences.

function _bernvec_n(u::T, n::Int) where {T<:Real}
    v = zeros(T, n+1)
    if iszero(u)
        v[1] = 1
        return v
    elseif isone(u)
        v[end]=1
        return v
    end
    p = (1 - u)^n
    v[1] = p
    @inbounds for k in 0:n-1
        p *= ((n - k) / (k + 1)) * (u / (1 - u))
        v[k+2] = p
    end
    return v
end

# ========
#   CDF
# ========

function _cdf(C::BetaCopula{d}, u) where {d}
    n = C.n
    # tablas por dimensión en el punto u
    CDFtab = ntuple(j -> Base.reverse(cumsum(Base.reverse(_bernvec_n(u[j], n))))[2:end], d)
    total = zero(eltype(first(CDFtab)))
    @inbounds for i in 1:n
        prod_term = one(total)
        @inbounds for j in 1:d
            prod_term *= CDFtab[j][C.ranks[j,i]]
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

function Distributions._logpdf(C::BetaCopula{d}, u::AbstractVector) where {d}
    n = C.n
    PDFtab = ntuple(j -> n .* _bernvec_n(u[j], n-1), d)
    dens = zero(eltype(first(PDFtab)))
    @inbounds for i in 1:n
        prod_term = one(dens)
        @inbounds for j in 1:d
            prod_term *= PDFtab[j][C.ranks[j,i]]
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
