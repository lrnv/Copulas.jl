"""
    MixedTail{T}, MixedCopula{d,T}

    MixedCopula{d}(θ)
    MixedCopula(d, θ)

Mixed extreme-value model with `θ ∈ [0,1]`.

In dimension two its Pickands dependence function is

```math
A(t)=1-\\theta t(1-t).
```

The original bivariate model is described by Tawn
[tawn1988bivariate](@cite).

For `d ≥ 2`, Copulas.jl uses the dimension-free extension

```math
\\ell_{\\mathrm{Mixed},\\theta}(x)
=
(1-\\theta)\\sum_{i=1}^d x_i
+
\\theta,\\ell_{\\mathrm{Galambos},1}(x).
```

This is a convex combination of the independence STDF and the multivariate
Galambos STDF with parameter one, hence it is a valid STDF in every supported
dimension. In `d=2` it reduces exactly to the historical Mixed Pickands model.

!!! note "Copulas.jl implementation derivation"
    The cited Tawn paper supports the original bivariate Mixed family and
    [galambos1975order](@cite) supports the negative-logistic component. The
    dimension-free convex-combination identity above is the extension derived
    and used in Copulas.jl; it is not attributed here as a formula from either
    source.

Special case:

* `θ = 0` returns `IndependentCopula(d)`.
"""
MixedTail, MixedCopula

struct MixedTail{T} <: OneParameterPickandsTail
    θ::T
    function MixedTail(θ)
        (0 ≤ θ ≤ 1+eps(θ)) || throw(ArgumentError("θ must be in [0,1], provided θ=$θ"))
        θ = clamp(θ, 0, 1)
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const MixedCopula{d,T} = ExtremeValueCopula{d, MixedTail{T}}
Distributions.params(tail::MixedTail) = (θ = tail.θ,)
_is_valid_in_dim(::MixedTail, d::Int) = d >= 2
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:MixedTail}, X::AbstractMatrix{T},) where {T<:Real}
    size(X, 1) == 2 || throw(DimensionMismatch("output must have two rows for a bivariate Mixed copula",))
    return _mixed_rand_multivariate!(rng, C.tail, X)
end
_unbound_params(::Type{<:MixedTail}, d, θ) = [log(θ.θ) - log1p(-θ.θ)]
_rebound_params(::Type{<:MixedTail}, d, α) = begin
    θ = 1 / (1 + exp(-α[1]))
    return (; θ)
end
_θ_bounds(::Type{<:MixedTail}, d) = (0.0, 1.0)

A(tail::MixedTail, t::Real) = tail.θ * t^2 - tail.θ * t + 1


# The bivariate Mixed model satisfies exactly
#
#   ℓ_Mixed(x) = (1-θ) * Σᵢ xᵢ + θ * ℓ_Galambos,α=1(x).
#
# This convex combination of stable tail dependence functions gives a
# dimension-free extension while reproducing the historical Pickands model
# exactly for d = 2.
@inline _mixed_galambos_tail(tail::MixedTail) = GalambosTail(one(tail.θ))

function ℓ(tail::MixedTail, x)
    θ = tail.θ
    return (one(θ) - θ) * sum(x) + θ * ℓ(_mixed_galambos_tail(tail), x)
end

function _ellpartial_signlog(tail::MixedTail, x, I::Tuple{Vararg{Int}},)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    θ = float(tail.θ)
    signg, logg = _ellpartial_signlog(_mixed_galambos_tail(tail), x, I,)

    signg == 0 && begin
        if length(I) == 1 && θ < 1
            return 1, log1p(-θ)
        end
        return 0, -Inf
    end

    if length(I) == 1
        logind = θ < 1 ? log1p(-θ) : -Inf
        logdep = log(θ) + logg
        return 1, LogExpFunctions.logaddexp(logind, logdep)
    end

    return signg, log(θ) + logg
end




function _mixed_rand_multivariate!(rng::Distributions.AbstractRNG, tail::MixedTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d >= 2 || throw(DimensionMismatch("MixedTail requires at least two output rows",))

    θ = Float64(tail.θ)
    Z = zeros(Float64, d, n)

    # Independent max-stable component with exponent
    # (1-θ) Σᵢ xᵢ.
    if θ < 1
        w = 1 - θ
        @inbounds for i in 1:d, col in 1:n
            Z[i, col] = w / Random.randexp(rng)
        end
    end

    # Galambos(1) max-stable component with exponent
    # θ ℓ_Galambos,1.
    if θ > 0
        Cgal = ExtremeValueCopula(d, GalambosTail(1.0))
        U = Random.rand(rng, Cgal, n)

        @inbounds for i in 1:d, col in 1:n
            candidate = θ / (-log(Float64(U[i, col])))
            if candidate > Z[i, col]
                Z[i, col] = candidate
            end
        end
    end

    @inbounds for i in 1:d, col in 1:n
        zi = Z[i, col]
        zi > 0 || throw(ArgumentError("invalid zero Fréchet value in MixedTail sampler",))
        X[i, col] = T(exp(-inv(zi)))
    end

    return X
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:MixedTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return _mixed_rand_multivariate!(rng, C.tail, X)
end

function dA(tail::MixedTail, t::Real)
    tt = _safett(t)
    θ = tail.θ

    return θ * (2tt - 1)
end

function d²A(tail::MixedTail, t::Real)
    θ = tail.θ

    return 2θ
end

_tau_Mixed(θ; kw...) = θ ≤ 0 ? 0.0 : θ ≥ 1 ? 1.0 : 1 + 4 * QuadGK.quadgk(t -> ((2θ*t - θ) / (θ*t^2 - θ*t + 1)) * t * (1-t), 0, 1; kw...)[1]
_rho_Mixed(θ; kw...) = θ ≤ 0 ? 0.0 : θ ≥ 1 ? 1.0 : 12 * QuadGK.quadgk(t -> inv((θ*t^2 - θ*t + 1 + 1)^2), 0, 1; kw...)[1] - 3

τ(C::ExtremeValueCopula{2,<:MixedTail}) = 8 / sqrt(C.tail.θ * (4 - C.tail.θ)) * atan( sqrt(C.tail.θ / (4 - C.tail.θ)) ) - 2
ρ(C::ExtremeValueCopula{2,<:MixedTail}) = -3 + 12/(8 - C.tail.θ) + 96 * atan(sqrt(C.tail.θ/(8 - C.tail.θ))) / (sqrt(C.tail.θ) * (8 - C.tail.θ)^(3/2))
β(C::ExtremeValueCopula{2,<:MixedTail}) = 2.0^(C.tail.θ / 2) - 1
λᵤ(C::ExtremeValueCopula{2,<:MixedTail}) = C.tail.θ / 2

τ⁻¹(::Type{<:ExtremeValueCopula{D,<:MixedTail} where D}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? 1 : _invmono(θ -> _tau_Mixed(θ) - τ; kw...)
τ⁻¹(::Type{<:MixedTail}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? 1 : _invmono(θ -> _tau_Mixed(θ) - τ; kw...)
ρ⁻¹(::Type{<:ExtremeValueCopula{D,<:MixedTail} where D}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? 1 : _invmono(θ -> _rho_Mixed(θ) - ρ; kw...)
β⁻¹(::Type{<:ExtremeValueCopula{D,<:MixedTail} where D}, beta) = 2 * log2(beta + 1)
λᵤ⁻¹(::Type{<:ExtremeValueCopula{D,<:MixedTail} where D}, λ) = 2λ
