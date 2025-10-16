"""
    GumbelBarnettGenerator{T}, GumbelBarnettCopula{d, T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelBarnettGenerator(θ)
    GumbelBarnettCopula(d,θ)

The Gumbel-Barnett copula is an archimdean copula with generator:

```math
    \\phi(t) = \\exp\\big( \\theta^{-1} (1 - e^{t}) \\big),\\quad 0 \\le \\theta \\le 1.
```

Special cases:
- When θ = 0, it is the IndependentCopula

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.437
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
GumbelBarnettGenerator, GumbelBarnettCopula

struct GumbelBarnettGenerator{T} <: AbstractUnivariateGenerator
    θ::T
    function GumbelBarnettGenerator(θ)
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const GumbelBarnettCopula{d, T} = ArchimedeanCopula{d, GumbelBarnettGenerator{T}}
Distributions.params(G::GumbelBarnettGenerator) = (θ = G.θ,)
function _unbound_params(::Type{<:GumbelBarnettGenerator}, d, θ)
    u = clamp(_find_critical_value_gumbelbarnett(d), 0, 1)
    return [atanh(2θ.θ/u - 1)]
end
function _rebound_params(::Type{<:GumbelBarnettGenerator}, d, α)
    u = clamp(_find_critical_value_gumbelbarnett(d), 0, 1)
    return (; θ = u*(tanh(α[1])+1)/2)
end
_θ_bounds(::Type{<:GumbelBarnettGenerator}, d) = (0.0, clamp(_find_critical_value_gumbelbarnett(d), 0.0, 1.0))

function _find_critical_value_gumbelbarnett(d::Integer)
    d == 2 && return 1.0
    d == 3  && return 0.380
    d == 4  && return 0.216
    d == 5  && return 0.145
    d == 6  && return 0.106
    d == 7  && return 0.082
    d == 8  && return 0.066
    d == 9  && return 0.055
    d == 10 && return 0.046

    C2 = binomial(d, 2)
    C3 = binomial(d, 3)
    C4 = binomial(d, 4)
    term1 = C2 / d
    discr = C2^2 - (2d / (d - 1)) * (C3 + 3*C4)
    if !(discr >= 0)
        return 0.046  # conservative fallback (d=10 value)
    end
    term2 = (d - 1)/d * sqrt(discr)
    lower_bound = (term1 + term2)
    return 1 / lower_bound
end

function max_monotony(G::GumbelBarnettGenerator)
    G.θ == 0 && return Inf
    # Check low dimensions via centralized thresholds
    for d in 3:1_000
        if G.θ > _find_critical_value_gumbelbarnett(d)
            return d-1
        end
    end
    return 1_000
end


ϕ(G::GumbelBarnettGenerator, t) = exp((1 - exp(t)) / G.θ)
ϕ⁽¹⁾(G::GumbelBarnettGenerator, t) = -exp((1 - exp(t)) / G.θ) * exp(t) / G.θ
ϕ⁻¹(G::GumbelBarnettGenerator, t) = log1p(-G.θ * log(t))
ϕ⁻¹⁽¹⁾(G::GumbelBarnettGenerator, t) = -G.θ / (t - G.θ * t * log(t))
function ϕ⁽ᵏ⁾(G::GumbelBarnettGenerator, k::Int, t)
    α = 1/G.θ    
    C = -α*exp(t)
    R = C * exp(α + C)
    k == 1 && return R
    return evalpoly(C, ntuple(i->Combinatorics.stirlings2(k, i), k)) * R
end


# See this htread ;: https://discourse.julialang.org/t/solving-for-transcendental-equation/131229/16


function _gumbelbarnett_tau(θ)
    iszero(θ) && return θ
    r, _ = QuadGK.quadgk(x -> (1-θ*log(x))  * log1p(-θ*log(x)) * x, 0, 1)
    return 1-4*r/θ
end

τ(G::GumbelBarnettGenerator) = _gumbelbarnett_tau(G.θ)
function τ⁻¹(::Type{<:GumbelBarnettGenerator}, τ)
    τ ≤ -0.3612 && return one(τ)
    τ ≥ 0 && return zero(τ)
    return Roots.find_zero(θ -> _gumbelbarnett_tau(θ) - τ, (0, 1))
end

# Edge utilities and robust bracketing
_GB_EPSA = 1e-12                 # separa de 0
_GB_EPSB = 1e-12                 # separa de 1
c_GB_TOLV = 1e-12                 # tolerancia en valor

# Internal grids to rescue bracketing if there are numerical problems
_GB_GRID_A = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2)
_GB_GRID_B = (1 - 1e-12, 1 - 1e-10, 1 - 1e-8, 1 - 1e-6, 1 - 1e-4, 0.999, 0.99, 0.95, 0.9)
function _rho_gumbelbarnett(θ)
    θ ≤ 0 && return zero(θ)
    r, _ = QuadGK.quadgk(z -> exp(-z)/(1+θ*z), 0, Inf)
    return r-1
end
ρ(G::GumbelBarnettGenerator) = _rho_gumbelbarnett(G.θ)
function ρ⁻¹(::Type{<:GumbelBarnettGenerator}, ρ)
    ρmin = _rho_gumbelbarnett(1 - _GB_EPSB)          # ≈ -0.266… 
    ρ ≤ ρmin && return one(ρ)
    ρ ≥ 0 && return zero(ρ)
    return Roots.find_zero(t -> _rho_gumbelbarnett(t) - ρ, (0, 1))
end