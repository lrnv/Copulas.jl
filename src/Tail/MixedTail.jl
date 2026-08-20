"""
    MixedTail{T}, MixedCopula{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    MixedCopula(θ)
    ExtremeValueCopula(2, MixedTail(θ))

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
MixedTail, MixedCopula

struct MixedTail{T} <: AbstractUnivariateTail2
    θ::T
    function MixedTail(θ)
        (0 ≤ θ ≤ 1+eps(θ)) || throw(ArgumentError("θ must be in [0,1], provided θ=$θ"))
        θ = clamp(θ, 0, 1)
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const MixedCopula{T} = ExtremeValueCopula{2, MixedTail{T}}
Distributions.params(tail::MixedTail) = (θ = tail.θ,)
_is_valid_in_dim(::MixedTail, d::Int) = d >= 2
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
@inline _mixed_galambos_tail(tail::MixedTail) =
    GalambosTail(one(tail.θ))

function ℓ(tail::MixedTail, x)
    θ = tail.θ
    return (one(θ) - θ) * sum(x) +
           θ * ℓ(_mixed_galambos_tail(tail), x)
end

function _ellpartial_signlog(
    tail::MixedTail,
    x,
    I::Tuple{Vararg{Int}},
)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    θ = float(tail.θ)
    signg, logg = _ellpartial_signlog(
        _mixed_galambos_tail(tail),
        x,
        I,
    )

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

_ellpartial_signlog(
    tail::MixedTail,
    x,
    I::AbstractVector{<:Integer},
) = _ellpartial_signlog(tail, x, Tuple(I))

function ellpartial(
    tail::MixedTail,
    x,
    I::Tuple{Vararg{Int}},
)
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    sign == 0 && return zero(float(first(x)))
    return sign * exp(logabs)
end

ellpartial(
    tail::MixedTail,
    x,
    I::AbstractVector{<:Integer},
) = ellpartial(tail, x, Tuple(I))

Distributions._logpdf(
    C::ExtremeValueCopula{d,<:MixedTail},
    u,
) where {d} = _ev_logpdf_from_partials(C, u)

# Resolve the intersection with the generic bivariate EV density.
Distributions._logpdf(
    C::ExtremeValueCopula{2,<:MixedTail},
    u,
) = _ev_logpdf_bivariate(C, u)

function _mixed_rand_multivariate!(
    rng::Distributions.AbstractRNG,
    tail::MixedTail,
    X::AbstractMatrix{T},
) where {T<:Real}
    d, n = size(X)
    d >= 2 || throw(DimensionMismatch(
        "MixedTail requires at least two output rows",
    ))

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
        zi > 0 || throw(ArgumentError(
            "invalid zero Fréchet value in MixedTail sampler",
        ))
        X[i, col] = T(exp(-inv(zi)))
    end

    return X
end

_rand_ev_multivariate!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:MixedTail},
    X::AbstractMatrix{T},
) where {d,T<:Real} =
    _mixed_rand_multivariate!(rng, C.tail, X)

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

τ(C::MixedCopula) = 8 / sqrt(C.tail.θ * (4 - C.tail.θ)) * atan( sqrt(C.tail.θ / (4 - C.tail.θ)) ) - 2
ρ(C::MixedCopula) = -3 + 12/(8 - C.tail.θ) + 96 * atan(sqrt(C.tail.θ/(8 - C.tail.θ))) / (sqrt(C.tail.θ) * (8 - C.tail.θ)^(3/2))
β(C::MixedCopula) = 2.0^(C.tail.θ / 2) - 1
λᵤ(C::MixedCopula) = C.tail.θ / 2

τ⁻¹(::Type{<:MixedCopula}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? 1 : _invmono(θ -> _tau_Mixed(θ) - τ; kw...)
ρ⁻¹(::Type{<:MixedCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? 1 : _invmono(θ -> _rho_Mixed(θ) - ρ; kw...)
β⁻¹(::Type{<:MixedCopula}, beta) = 2 * log2(beta + 1)
λᵤ⁻¹(::Type{<:MixedCopula}, λ) = 2λ
