"""
    ClaytonGenerator(θ)
    ClaytonCopula{d}(θ)
    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)`` (with the independence case as the limit ``\\theta\\to 0``). It is an Archimedean copula with generator

```math
\\phi(t) = \\left(1 + \\theta t\\right)^{-1/\\theta}
```

with the continuous extension ``\\phi(t) = e^{-t}`` at ``\\theta = 0``.

Special cases (for the copula in dimension ``d``):
- ``\\theta = -1/(d-1)`` gives the lower Fréchet–Hoeffding bound.
- ``\\theta = 0`` gives the independence copula.
- ``\\theta = \\infty`` gives the upper Fréchet–Hoeffding bound.

Positive parameters produce lower-tail dependence but no upper-tail
dependence. Negative parameters are valid only up to `d ≤ 1 - 1/θ`; equality
introduces a singular component, so ordinary-density likelihood reasoning
requires care at that boundary.

See also: [`Generator`](@ref), [`ArchimedeanCopula`](@ref), [`λₗ`](@ref),
[`λᵤ`](@ref), [`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
ClaytonGenerator, ClaytonCopula

Paramorph.@paramorph T struct ClaytonGenerator{T<:Real} <: AbstractUnivariateGenerator
    θ::closed_lower(get(context, :lower, -one(T)))
end
ClaytonGenerator(θ::Integer) = ClaytonGenerator(float(θ))
const ClaytonCopula{d, T} = ArchimedeanCopula{d, ClaytonGenerator{T}}
@inline function limit_kind(G::ClaytonGenerator, ::Val{d}) where {d}
    iszero(G.θ) && return Π_LIMIT
    isinf(G.θ) && return M_LIMIT
    d == 2 && G.θ == -1 && return W_LIMIT
    return NO_LIMIT
end

max_monotony(G::ClaytonGenerator) = G.θ >= 0 ? Inf : (1 - 1/G.θ)
archimedean_measure_style(G::ClaytonGenerator, ::Val{d}) where {d} =
    (G.θ == -1 / (d - 1) || !isfinite(G.θ)) ? NonAbsolutelyContinuousMeasure() : AbsolutelyContinuousMeasure()
# The generator and its derivatives in `log1p`/`expm1` form. The power forms
# `(1 + θt)^(-1/θ)` and `(t^(-θ) - 1)/θ` cancel to `1` and `0` once `θ` drops
# below `eps`, which puts every Clayton CDF value at `1`; the logarithmic forms
# are exact there and reach the `θ = 0` branch continuously. A non-finite `θ`
# keeps the power form, whose limits are the ones it always had.
ϕ(  G::ClaytonGenerator, t) = iszero(G.θ) ? exp(-t) : !isfinite(G.θ) ? max(1+G.θ*t,zero(t))^(-1/G.θ) : (1+G.θ*t) ≤ 0 ? zero(1+G.θ*t) : exp(-log1p(G.θ*t)/G.θ)
ϕ⁻¹(G::ClaytonGenerator, t) = iszero(G.θ) ? -log(t) : !isfinite(G.θ) ? (t^(-G.θ)-1)/G.θ : expm1(-G.θ*log(t))/G.θ
ϕ⁽¹⁾(G::ClaytonGenerator, t) = iszero(G.θ) ? -exp(-t) : (1+G.θ*t) ≤ 0 ? zero(1+G.θ*t) : !isfinite(G.θ) ? - (1+G.θ*t)^(-1/G.θ -1) : -exp(-(1+G.θ)*log1p(G.θ*t)/G.θ)
ϕ⁻¹⁽¹⁾(G::ClaytonGenerator, t) = iszero(G.θ) ? -inv(t) : -t^(-G.θ-1)
ϕ⁽ᵏ⁾(G::ClaytonGenerator, k::Int, t) = (1+G.θ*t) ≤ 0 ? zero(1+G.θ*t) : iszero(G.θ) ? (-1)^k * exp(-t) : !isfinite(G.θ) ? (1 + G.θ * t)^(-1/G.θ - k) * prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1) : exp(-(1+k*G.θ)*log1p(G.θ*t)/G.θ) * prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1)
ϕ⁽ᵏ⁾⁻¹(G::ClaytonGenerator, k::Int, t; start_at=t) = iszero(G.θ) ? -log(abs(t)) : !isfinite(G.θ) ? ((t / prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1))^(1/(-1/G.θ - k)) -1)/G.θ : expm1(-G.θ*log(t / prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1))/(1+k*G.θ))/G.θ

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

@inline _clayton_primal(x) = x
@inline _clayton_primal(x::ForwardDiff.Dual) = _clayton_primal(ForwardDiff.value(x))

function _clayton_independence_logpdf(θ, u, ::Val{d}) where {d}
    T = θ + zero(eltype(u))
    L1 = zero(T)
    L2 = zero(T)
    L3 = zero(T)
    @inbounds for t in u
        zero(t) < t < one(t) || return oftype(T, -Inf)
        lt = log(t)
        L1 += lt
        L2 += lt * lt
        L3 += lt * lt * lt
    end

    # If q(θ) = Σ(expm1(-θ log uᵢ)), write
    # log1p(q(θ)) = b₁θ + b₂θ² + b₃θ³ + O(θ⁴). The singular factor
    # -(1/θ + d) has a removable singularity, and these coefficients give the
    # log-density through second order. Keeping θ in the polynomial preserves
    # first and second derivatives for ForwardDiff at the independence point.
    b2 = (L2 - L1 * L1) / 2
    b3 = -L1 * L1 * L1 / 3 + L1 * L2 / 2 - L3 / 6
    k1 = oftype(T, d * (d - 1)) / 2
    k2 = -oftype(T, d * (d - 1) * (2d - 1)) / 12
    c1 = k1 + (d - 1) * L1 - b2
    c2 = k2 - b3 - d * b2
    return θ * (c1 + θ * c2)
end

function _archimedean_logpdf(C::ClaytonCopula{d}, u) where {d}
    θ = C.G.θ
    T = θ + zero(eltype(u))

    # The primal independence value has a removable singularity in the closed
    # form below. Use its second-order expansion for Dual values whose primal is
    # exactly zero so gradient/Hessian based fitting can start at independence.
    # A plain scalar θ == 0 is routed through Π_LIMIT before reaching this method.
    iszero(_clayton_primal(θ)) && return _clayton_independence_logpdf(θ, u, Val(d))

    # S1 is Σ (tᵢ^(-θ) - 1), accumulated through `expm1` so that it does not
    # cancel below eps; the density's last factor is (S1 + 1)^(-1/θ - d).
    S1 = zero(T)
    S2 = zero(eltype(u))
    @inbounds for t in u
        zero(t) < t < one(t) || return oftype(T, -Inf)
        lt = log(t)
        S1 += expm1(-θ * lt)
        S2 += lt
    end

    if θ < 0 && S1 < -1
        return oftype(T, -Inf)
    end

    S1 == -1 && return oftype(T, -Inf)
    logcoef = zero(T)
    @inbounds for k in 1:(d - 1)
        logcoef += log1p(k * θ)
    end

    return logcoef - (θ + 1) * S2 + (-inv(θ) - d) * log1p(S1)
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
