"""
    JoeGenerator{T}, JoeCopula{d, T}

Fields:
  - θ::Real - parameter

Constructor

    JoeGenerator(θ)
    JoeCopula(d,θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\big(1 - e^{-t}\\big)^{1/\\theta}.
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ = ∞, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
JoeGenerator, JoeCopula

struct JoeGenerator{T} <: AbstractUnivariateFrailtyGenerator
    θ::T
    function JoeGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const JoeCopula{d, T} = ArchimedeanCopula{d, JoeGenerator{T}}
frailty(G::JoeGenerator) = Sibuya(1/G.θ)
Distributions.params(G::JoeGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:JoeGenerator}, d, θ) = [log(θ.θ - 1)]
_rebound_params(::Type{<:JoeGenerator}, d, α) = (; θ = 1 + exp(α[1]))
_θ_bounds(::Type{<:JoeGenerator}, d) = (1, Inf)

ϕ(  G::JoeGenerator, t) = 1-(-expm1(-t))^(1/G.θ)
ϕ⁻¹(G::JoeGenerator, t) = -log1p(-(1-t)^G.θ)
ϕ⁽¹⁾(G::JoeGenerator, t) = (-expm1(-t))^(1/G.θ) / (G.θ - G.θ * exp(t))
function ϕ⁽ᵏ⁾(G::JoeGenerator, d::Int, t)
    # TODO: test if this ϕ⁽ᵏ⁾ is really more 'efficient' than the default one, 
    # as we already saw that for the Gumbel is wasn't the case. 
    α = 1 / G.θ
    x = exp(-t)
    y = -expm1(-t)
    r = x/y
    P_d_α = sum(Combinatorics.stirlings2(d, k) * (SpecialFunctions.gamma(k - α) / SpecialFunctions.gamma(1 - α)) * r^(k-1) for k in 1:d)
    return (-1)^d * α * (x / y^(1 - α)) * P_d_α
end
function ϕ⁻¹⁽¹⁾(G::JoeGenerator, t)
    return -(G.θ * (1 - t)^(G.θ - 1)) / (1 - (1 - t)^G.θ)
end

function ϕ⁽ᵏ⁾⁻¹(G::JoeGenerator, k::Int, y; start_at=y)
    k == 1 || return @invoke ϕ⁽ᵏ⁾⁻¹(G::Generator, k, y; start_at=start_at)
    θ, yy = promote(float(G.θ), float(y))
    yy ≤ zero(yy) || throw(DomainError(y, "The first generator derivative is non-positive."))
    m = -yy
    iszero(m) && return typeof(m)(Inf)
    isinf(m) && return zero(m)

    α, logm = inv(θ), log(m)
    logα = log(α)
    f(z) = logα - LogExpFunctions.log1pexp(z) - (α - one(z)) * LogExpFunctions.log1pexp(-z) - logm
    df(z) = α - one(z) - α * LogExpFunctions.logistic(z)
    z = logm ≥ logα ? (logm - logα) / (α - one(α)) : logα - logm
    lo, hi = min(z, -one(z)), max(z, one(z))
    while f(lo) < zero(z); lo = 2lo - one(z); end
    while f(hi) > zero(z); hi = 2hi + one(z); end

    z = clamp(z, lo, hi)
    for _ in 1:(precision(typeof(z)) + 16)
        fz = f(z)
        abs(fz) ≤ 16eps(typeof(z)) * (one(z) + abs(logm)) && break
        candidate = z - fz / df(z)
        (!isfinite(candidate) || !(lo < candidate < hi)) && (candidate = lo + (hi - lo) / 2)
        if f(candidate) > zero(z); lo = candidate else hi = candidate end
        z = candidate
    end
    return LogExpFunctions.log1pexp(z)
end

_joe_tau(θ) =  1 - 4sum(1/(k*(2+k*θ)*(θ*(k-1)+2)) for k in 1:1000)
τ(G::JoeGenerator) = _joe_tau(G.θ)
function τ⁻¹(::Type{<:JoeGenerator}, τ)
    l, u = one(τ), τ * Inf
    τ ≤ 0 && return l
    τ ≥ 1 && return u
    τ = clamp(τ, 0, 1)
    return Roots.find_zero(θ -> _joe_tau(θ) - τ, (l, u))
end

_rho_joe(θ) = @invoke ρ(JoeCopula(2, θ)::Copula)
ρ(G::JoeGenerator) = _rho_joe(G.θ)
function ρ⁻¹(::Type{<:JoeGenerator}, ρ)
    l, u = one(ρ), ρ * Inf
    ρ ≤ 0 && return l
    ρ ≥ 1 && return u
    return Roots.find_zero(θ -> _rho_joe(θ) - ρ, (1, Inf))
end
