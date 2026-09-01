"""
    ClaytonGenerator{T}, ClaytonCopula{d, T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonGenerator(θ)
    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)`` (with the independence case as the limit ``\\theta\\to 0``). It is an Archimedean copula with generator

```math
\\phi(t) = \\left(1 + \\theta t\\right)^{-1/\\theta}
```

with the continuous extension ``\\phi(t) = e^{-t}`` at ``\\theta = 0``.

Special cases (for the copula in dimension ``d``):
- When ``\\theta = -1/(d-1)``, it is the WCopula (Lower Fréchet–Hoeffding bound)
- When ``\\theta = 0``, it is the IndependentCopula
- When ``\\theta = \\infty``, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
ClaytonGenerator, ClaytonCopula

struct ClaytonGenerator{T} <: AbstractUnivariateGenerator
    θ::T
    function ClaytonGenerator(θ)
        θ >= -1 || throw(ArgumentError("Theta must be greater than or equal to -1"))
        θf = float(θ)
        return new{typeof(θf)}(θf)
    end
end
const ClaytonCopula{d, T} = ArchimedeanCopula{d, ClaytonGenerator{T}}
@inline function limit_kind(G::ClaytonGenerator, ::Val{d}) where {d}
    isinf(G.θ) && return M_LIMIT
    d == 2 && G.θ == -1 && return W_LIMIT
    return NO_LIMIT
end

Distributions.params(G::ClaytonGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:ClaytonGenerator}, d, θ) = [log(θ.θ + 1/(d-1))] # θ > -1/(d-1) ⇒ θ+1/(d-1)>0
_rebound_params(::Type{<:ClaytonGenerator}, d, α) = (; θ = exp(α[1]) - 1/(d-1))
_θ_bounds(::Type{<:ClaytonGenerator}, d) = (-1/(d-1), Inf)

max_monotony(G::ClaytonGenerator) = G.θ >= 0 ? Inf : (1 - 1/G.θ)
archimedean_measure_style(G::ClaytonGenerator, ::Val{d}) where {d} =
    (G.θ == -1 / (d - 1) || !isfinite(G.θ)) ? NonAbsolutelyContinuousMeasure() : AbsolutelyContinuousMeasure()
ϕ(  G::ClaytonGenerator, t) = iszero(G.θ) ? exp(-t) : max(1+G.θ*t,zero(t))^(-1/G.θ)
ϕ⁻¹(G::ClaytonGenerator, t) = iszero(G.θ) ? -log(t) : (t^(-G.θ)-1)/G.θ
ϕ⁽¹⁾(G::ClaytonGenerator, t) = iszero(G.θ) ? -exp(-t) : (1+G.θ*t) ≤ 0 ? 0 : - (1+G.θ*t)^(-1/G.θ -1)
ϕ⁻¹⁽¹⁾(G::ClaytonGenerator, t) = iszero(G.θ) ? -inv(t) : -t^(-G.θ-1)
ϕ⁽ᵏ⁾(G::ClaytonGenerator, k::Int, t) = (1+G.θ*t) ≤ 0 ? 0 : iszero(G.θ) ? (-1)^k * exp(-t) : (1 + G.θ * t)^(-1/G.θ - k) * prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1)
ϕ⁽ᵏ⁾⁻¹(G::ClaytonGenerator, k::Int, t; start_at=t) = iszero(G.θ) ? -log(abs(t)) : ((t / prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1))^(1/(-1/G.θ - k)) -1)/G.θ

# Closed-form edge-composition override for a Clayton-over-Clayton nesting. Overrides the
# default `composition_taylor` hook (nested/NestedArchimedeanDensity.jl) by dispatch, and
# returns the Taylor coefficients [h'(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!] of the inner→outer change of
# variables h(t) = ϕ⁻¹_outer(ϕ_inner(t)). With ϕ_θ(t)=(1+θt)^(-1/θ) and ϕ⁻¹_θ(u)=(u^(-θ)-1)/θ,
# the link is h(t) = ((1+θ_in·t)^r − 1)/θ_out, r = θ_out/θ_in — a reparametrised power map
# whose coefficients are a generalized binomial series, so it never touches the (ill-
# conditioned) high-order derivatives of the inverse. NOTE: the θ live INSIDE ϕ here, so the
# expansion base is 1+θ_in·t₀ (NOT 1+t₀); θ promotes into T so Float64/BigFloat stay exact.
function composition_taylor(outer::ClaytonGenerator, inner::ClaytonGenerator, t₀::T, d::Int) where {T}
    if iszero(outer.θ) || iszero(inner.θ) || !isfinite(outer.θ) || !isfinite(inner.θ)
        return invoke(
            composition_taylor,
            Tuple{Generator,Generator,T,Int},
            outer,
            inner,
            t₀,
            d,
        )
    end

    θ_out = T(outer.θ)
    θ_in  = T(inner.θ)
    r     = θ_out / θ_in
    base  = 1 + θ_in * t₀
    h = Vector{T}(undef, d)
    binom = one(T)                                     # generalized binomial C(r,k), incremental
    θ_in_pow = one(T)                                  # θ_in^k
    for k in 1:d
        binom    *= (r - (k - 1)) / k                  # C(r,k) = C(r,k-1)·(r-k+1)/k
        θ_in_pow *= θ_in
        h[k] = (θ_in_pow / θ_out) * binom * base^(r - k)
    end
    return h
end

τ(G::ClaytonGenerator) = ifelse(isfinite(G.θ), G.θ/(G.θ+2), 1)
τ⁻¹(::Type{<:ClaytonGenerator},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))
# For θ > 0, the Clayton frailty is θY with Y ~ Gamma(1/θ, 1), so
# Wₑ⁻¹(ϕ) = Gamma(d, 1)/(θY) = BetaPrime(d, 1/θ)/θ. This exact
# representation works for real orders and avoids numerical quadrature for
# repeatedly evaluated Liouville radials and margins.

# Negative Clayton generators, on the other hand,
# have no frailty representation: integer orders retain their
# dedicated finite-support ClaytonWilliamsonDistribution, while non-integer
# orders fall back to the generic exact beta reduction from the next integer
# Williamson order.
function 𝒲₋₁(G::ClaytonGenerator, d::Integer)
    iszero(G.θ) && return Distributions.Gamma(d, 1)
    G.θ > 0 && isfinite(G.θ) && return Distributions.BetaPrime(d, inv(G.θ))*inv(G.θ)
    G.θ <= 0 && return ClaytonWilliamsonDistribution(G.θ, d)
    return invoke(𝒲₋₁, Tuple{Generator,Integer}, G, d)
end
function 𝒲₋₁(G::ClaytonGenerator, d::Real)
    iszero(G.θ) && return Distributions.Gamma(d, 1)
    G.θ > 0 && isfinite(G.θ) && return Distributions.BetaPrime(d, inv(G.θ))*inv(G.θ)
    return invoke(𝒲₋₁, Tuple{Generator,Real}, G, d)
end


frailty(G::ClaytonGenerator) =
    iszero(G.θ) ? Distributions.Dirac(1.0) :
    G.θ > 0 && isfinite(G.θ) ? Distributions.Gamma(inv(G.θ), G.θ) : nothing

function _archimedean_cdf(C::ClaytonCopula{d}, u) where {d}
    return @invoke _archimedean_cdf(C::ArchimedeanCopula, u)
end

function _archimedean_logpdf(C::ClaytonCopula{d,TG}, u) where {d,TG<:ClaytonGenerator}
    # Check if all elements are in (0,1) and if θ < 0, check the sum condition

    iszero(C.G.θ) && return @invoke _archimedean_logpdf(C::ArchimedeanCopula, u)

    if !all(0 .< u .< 1) || (C.G.θ < 0 && sum(u .^ -(C.G.θ)) < (d - 1))
        return eltype(u)(-Inf)
    end

    θ = C.G.θ
    # Compute the sum of transformed variables
    S1 = sum(t ^ (-θ) for t in u)
    S2 = sum(log(t) for t in u)
    # Compute the log of the density according to the explicit formula for Clayton copula
    # See McNeil & Neslehova (2009), eq. (13)
    S1==d-1 && return eltype(u)(-Inf)
    return log(θ + 1) * (d - 1) - (θ + 1) * S2 + (-1 / θ - d) * log(S1 - d + 1)
end

ρ(G::ClaytonGenerator) =
    isinf(G.θ) ? 1 :
    iszero(G.θ) ? 0 :
    @invoke ρ(ArchimedeanCopula(2, G)::Copula)

# Inverse ρ → θ for Clayton (without trimming to [0,1])
function ρ⁻¹(::Type{<:ClaytonGenerator}, ρ̂; atol=1e-10)
    _ρ = float(ρ̂)
    if isapprox(_ρ, 0.0; atol=1e-14)
        return 0.0
    end
    _ρ >= 1 && return Inf
    _ρ <= -1 && return -1.0

    f(θ) = ρ(ClaytonGenerator(θ)) - _ρ
    if _ρ < 0
        bracket = (-1 + sqrt(eps(Float64)), 0.0)
    else
        # Spearman's rho increases to one with θ. Grow the upper endpoint
        # until it brackets the requested value instead of relying on a
        # secant step, which is fragile for strongly dependent samples.
        upper = 1.0
        while f(upper) < 0
            upper *= 2
        end
        bracket = (0.0, upper)
    end
    return Roots.find_zero(f, bracket, Roots.Brent(); xatol=atol, rtol=0)
end
