"""
    LogTail{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    LogCopula(θ)
    ExtremeValueCopula(2, LogTail(θ))

The (bivariate) Mixed extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\theta t^2 - \\theta t + 1, \\quad t \\in [0,1].
```
Special cases:

* θ = 0 ⇒ IndependentCopula

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct LogTail{T} <: Tail2
    θ::T
    function LogTail(θ)
        !(1 <= θ) && throw(ArgumentError(" The param θ must be in [1, ∞)"))
        θ == 1 && return NoTail()
        isinf(θ) && return MTail()
        θ, _ = promote(θ, 1.0)
        return new{typeof(θ)}(θ)
    end
end

const LogCopula{T} = ExtremeValueCopula{2, LogTail{T}}
LogCopula(θ) = ExtremeValueCopula(2, LogTail(θ))
Distributions.params(tail::LogTail) = (tail.θ,)
_is_valid_in_dim(::LogTail, d::Int) = (d >= 2)

"""
    A(tail::LogTail, ω::NTuple{d,Real}) where d

Multi-dimensional Pickands dependence function on the simplex Δ_{d-1} for the
logistic (a.k.a. Gumbel) extreme-value model with parameter θ ≥ 1.

    A(ω) = (∑_{i=1}^d ω_i^θ)^{1/θ},   ω_i ≥ 0,  ∑ ω_i = 1.

We implement this in a numerically stable manner using a log-sum-exp style
aggregation on the log(ω_i) scale to mitigate underflow when some ω_i are very
small.
"""
function A(tail::LogTail, ω::NTuple{d,<:Real}) where {d}
    θ = tail.θ
    @inbounds begin
        # Handle potential exact zeros (allowed at the boundary of the simplex)
        # ω_i^θ = 0 if ω_i == 0 (θ ≥ 1). If all but one entry are zero the sum is 1.
        # Fast path: if one coordinate is 1 (degenerate vertex) return 1.
        for ωi in ω
            if ωi == 1.0
                return one(ωi)
            end
        end
        # Collect scaled logs; skip zeros to avoid -Inf + later exp
        logs = similar(ntuple(_->0.0, d))  # temporary tuple-like container
        mx = -Inf
        nz = 0
        for i in 1:d
            ωi = ω[i]
            if ωi > 0
                val = θ * log(ωi)
                logs = Base.setindex(logs, val, i)
                if val > mx; mx = val; end
                nz += 1
            else
                logs = Base.setindex(logs, -Inf, i)
            end
        end
        nz == 0 && return one(eltype(ω)) # degenerate (should not happen if ∑ ω_i =1)
        s = 0.0
        for i in 1:d
            li = logs[i]
            @inbounds if isfinite(li)
                s += exp(li - mx)
            end
        end
        return exp((mx + log(s)) / θ)
    end
end


# Placeholder for future optimized logistic sampler (Dirichlet / positive stable approach)
# Currently falls back to generic (inverse Rosenblatt) sampler above. Once validated
# we can replace by an O(d) method.

function Distributions._logpdf(C::ExtremeValueCopula{d, LogTail{Tθ}}, u) where {d, Tθ}
    # Analytic logistic (multivariate Gumbel / LogTail) pdf for d ≥ 3
    # ℓ(x) = (∑ x_i^θ)^{1/θ}, θ = C.tail.θ ≥ 1.
    # Density: c(u) = exp(-ℓ(x)) * (1/∏ u_i) * (∏ x_i^{θ-1}) * (∑ x_i^θ)^{1/θ - d} * A_d(θ)
    # where A_d(θ) obtained from Bell-type recursion:
    #   F_k(θ) = falling factorial θ(θ-1)…(θ-k+1)
    #   A[0]=1; A[n] = Σ_{k=1}^n (-1) * F_k(θ) * A[n-k] * binomial(n-1,k-1).
    # For θ=1 (independence), return logpdf = 0.
    # We guard against boundary (any u_i≈1 ⇒ x_i≈0). If θ>1 and some x_i == 0 ⇒ density → 0.
    

    d == 2 && return @invoke Distributions._logpdf(C::ExtremeValueCopula{2, LogTail{Tθ}}, u)
    θ = C.tail.θ
    # Domain check
    @inbounds for ui in u
        (0.0 < ui < 1.0) || return -Inf
    end
    if θ == 1
        # Independence (limit case)
        return 0.0
    end
    x = @inbounds (-log.(u))
    # If any x_i == 0 (u_i==1) density -> 0
    any(iszero, x) && return -Inf
    # Core sums
    xθ = map(y -> y^θ, x)
    s = zero(eltype(xθ))
    @inbounds for v in xθ; s += v; end
    s == 0 && return -Inf
    ℓx = s^(1/θ)
    # product of x_i^{θ-1}
    prod_r = zero(eltype(x)) + 1
    @inbounds for xi in x
        prod_r *= xi^(θ - 1)
    end
    # Recurrence for A_d(θ)
    # falling factorial cache
    F = Vector{eltype(x)}(undef, d)
    F[1] = θ
    @inbounds for k in 2:d
        F[k] = F[k-1] * (θ - (k - 1))
    end
    A = Vector{eltype(x)}(undef, d+1)
    A[1] = one(eltype(x)) # A[1] represents A_0
    # Shifted indexing: A[n+1] stores A_n
    for n in 1:d
        acc = zero(eltype(x))
        @inbounds for k in 1:n
            w_k = F[k]
            acc += (-1) * w_k * A[n - k + 1] * binomial(n - 1, k - 1)
        end
        A[n+1] = acc
    end
    A_dθ = A[d+1]
    A_dθ <= 0 && return @invoke Distributions._logpdf(C::ExtremeValueCopula{d, Tail}, u) # fallback to AD if numerically unstable
    logc = -ℓx - sum(log.(u)) + (θ - 1) * sum(log.(x)) + (1/θ - d) * log(s) + log(A_dθ)
    return logc
end

## ---------------------------------------------------------------------------
## Specialized sampler for logistic (LogTail) extreme value copula (any d)
## Based on spectral measure representation: If E_i ~ Exp(1) independent and
## G_i ~ Gamma(1/θ, 1) then set W_i = G_i / Σ G_i; define S = (Σ E_i / W_i)^{1/θ}.
## Then U_i = exp(-(S * (E_i / W_i)^{1/θ})) has the logistic EV copula with θ.
## (Derivation aligns with standard spectral construction for multivariate Gumbel.)
## ---------------------------------------------------------------------------
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d, LogTail{Tθ}}, X::AbstractMatrix{T}) where {d, Tθ, T<:Real}
    @assert size(X,1) == d
    θ = C.tail.θ
    invθ = 1/θ
    n = size(X,2)
    @inbounds for col in 1:n
        # Draw Dirichlet(1/θ,...,1/θ) via Gamma
        gsum = zero(Float64)
        for i in 1:d
            gi = rand(rng, Distributions.Gamma(invθ))
            X[i,col] = gi
            gsum += gi
        end
        # Normalize to W_i
        for i in 1:d
            X[i,col] /= gsum
        end
        # Exponential draws
        total = zero(Float64)
        for i in 1:d
            Ei = rand(rng)
            while Ei <= 0.0; Ei = rand(rng); end # ensure >0
            # store temporarily Ei in-place scaled later; reuse column
            # Compute contribution: Ei / W_i
            Wi = X[i,col]
            Xi = Ei / Wi
            X[i,col] = Xi  # hold Xi
            total += Xi
        end
        S = total^invθ
        for i in 1:d
            Xi = X[i,col]
            # Xi currently = Ei / W_i
            X[i,col] = exp(-(S * Xi^invθ))
        end
    end
    return X
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d, LogTail{Tθ}}, x::AbstractVector{T}) where {d, Tθ, T<:Real}
    θ = C.tail.θ; invθ = 1/θ
    # Dirichlet via gamma
    gsum = 0.0
    for i in 1:d
        gi = rand(rng, Distributions.Gamma(invθ))
        x[i] = gi
        gsum += gi
    end
    for i in 1:d
        x[i] /= gsum
    end
    total = 0.0
    for i in 1:d
        Ei = rand(rng)
        while Ei <= 0.0; Ei = rand(rng); end
        Wi = x[i]
        Xi = Ei / Wi
        x[i] = Xi
        total += Xi
    end
    S = total^invθ
    for i in 1:d
        Xi = x[i]
        x[i] = exp(-(S * Xi^invθ))
    end
    return x
end


## ---------------------------------------------------------------------------
## Logistic (LogTail) multi-dimensional distortion specialization (p=1)
## U_i | U_j = u_j has cdf F_{i|j}(u) = ∂_j C(u_i, u_j, 1, ...,1)/∂_j C(1, u_j,1,...,1)
## For an extreme-value copula C(u) = exp(-ℓ(-log u)), with ℓ homogeneous of
## order 1: ∂_j C = C * ( (∂_j ℓ)/u_j ). When fixing other arguments at 1,
## x_k = -log 1 = 0 removes those coordinates. For the logistic model
## ℓ(x) = (x_1^θ + x_2^θ)^{1/θ} if only coords i,j vary. We thus reduce to
## the bivariate distortion with effective 2D tail (same θ), so we can reuse
## the existing fast BivEVDistortion logic by mapping indices.
## ---------------------------------------------------------------------------

@inline function DistortionFromCop(C::ExtremeValueCopula{d, LogTail{Tθ}}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, i::Int) where {d, Tθ}
    # For p=1 we only need a bivariate slice involving coordinates (i, j)
    j = js[1]
    if i == j
        throw(ArgumentError("Cannot build distortion for identical conditioned and target index"))
    end
    # Reuse the existing BivEVDistortion with the same tail; orientation matters:
    # BivEVDistortion expects j∈{1,2} as which coordinate is conditioned.
    # We map (i,j) in original d-D space to a virtual 2D with ordering (free, conditioned).
    # If original conditioned index is j, we set j_virtual = 2.
    return BivEVDistortion(C.tail, Int8(2), float(uⱼₛ[1]))
end


##### Special binding for dim 2 
## ---------------------------------------------------------------------------
## Bivariate specializations (for performance / numerical stability)
## ---------------------------------------------------------------------------
A(tail::LogTail, t::Real) = begin
    θ = tail.θ
    return exp(LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t)) / θ)
end
function dA(tail::LogTail, t::Real)
    θ = tail.θ

    # B = t^θ + (1-t)^θ
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    Bpow = exp((1 - θ) / θ * logB)  # B^((1-θ)/θ)

    # D = t^(θ-1) - (1-t)^(θ-1)
    logt = (θ - 1) * log(t)
    log1mt = (θ - 1) * log1p(-t)
    # carrefull for cancellations
    if logt > log1mt
        D = exp(logt) - exp(log1mt)  # no cancellation here. 
    else
        D = exp(log1mt) * (expm1(logt - log1mt))
    end

    return Bpow * D
end
function d2A(tail::LogTail, t::Real)
    θ = tail.θ

    # B = t^θ + (1-t)^θ
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    B = exp(logB)

    # D = t^(θ-1) - (1-t)^(θ-1)
    logt = (θ - 1) * log(t)
    log1mt = (θ - 1) * log1p(-t)
    if logt > log1mt
        D = exp(logt) - exp(log1mt)
    else
        D = exp(log1mt) * (expm1(logt - log1mt))
    end

    # B' = θ*D
    Bp = θ * D

    # E = (θ-1)*(t^(θ-2) + (1-t)^(θ-2))
    logt2 = (θ - 2) * log(t)
    log1mt2 = (θ - 2) * log1p(-t)
    # lets avoid unstable additions
    if logt2 > log1mt2
        E = (θ - 1) * (exp(logt2) + exp(log1mt2))
    else
        E = (θ - 1) * (exp(log1mt2) * (1 + exp(logt2 - log1mt2)))
    end

    term1 = ((1 - θ) / θ) * exp((1 - 2θ) / θ * logB) * Bp * D
    term2 = exp((1 - θ) / θ * logB) * E

    return term1 + term2
end

