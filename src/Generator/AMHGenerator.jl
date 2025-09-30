"""
    AMHGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    AMHGenerator(θ)
    AMHCopula(d,θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t} - \\theta}.
```

Special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct AMHGenerator{T} <: AbstractUnivariateGenerator
    θ::T
    function AMHGenerator(θ)
        if (θ < -1) || (θ > 1)
            throw(ArgumentError("Theta must be in [-1,1), you provided $θ."))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const AMHCopula{d, T} = ArchimedeanCopula{d, AMHGenerator{T}}
AMHCopula(d, θ::Real) = ArchimedeanCopula(d, AMHGenerator(θ))
AMHCopula(d; θ::Real) = AMHCopula(d, θ)
Distributions.params(G::AMHGenerator) = (θ = G.θ,)
_example(::Type{<:AMHCopula}, d) = AMHCopula(d, 0.1)
function _unbound_params(CT::Type{<:AMHCopula}, d, θ)
    l =  _find_critical_value_amh(d, step=1e-7)
    [log(θ.θ - l) - log(1-l)]
end
function _rebound_params(CT::Type{<:AMHCopula}, d, α)
    l =  _find_critical_value_amh(d, step=1e-7)
    (; θ = (exp(α[1]) + l) / (exp(α[1]) + 1))
end
function _find_critical_value_amh(k; step=1e-7)
    # Return the threshold θ_k such that “θ < θ_k ⇒ max_monotony returns k-1”.
    # This unifies analytic and numeric thresholds and falls back to a
    # numerical search via PolyLog for large k.
    k == 2  && return -1.0
    k == 3  && return sqrt(3) - 2
    k == 4  && return -5 + 2*sqrt(6)
    k == 5  && return -13/2 - sqrt(105)/2 + (sqrt(2)/2) * sqrt(13*sqrt(105) + 135)
    k == 6  && return -14 - 3 * sqrt(15) + sqrt(6) * sqrt(14 * sqrt(15) + 55)
    k == 7  && return -0.00914869999999993
    k == 8  && return -0.004376199999998468
    k == 9  && return -0.002121400000000042
    k == 10 && return -0.0010375999999997928
    k == 11 && return -0.0005105999999999994
    k == 12 && return -0.00025240000000000527
    k == 13 && return -0.0001252000000000022
    k == 14 && return -6.220000000000067e-5
    k == 15 && return -3.099999999999991e-5
    k == 16 && return -1.5500000000000048e-5
    k == 17 && return -7.699999999999994e-6
    k == 18 && return -3.839999999999973e-6
    k == 19 && return -1.9199999999999918e-6
    k == 20 && return -9.600000000000008e-7

    x = 0.0
    while x > -1
        PolyLog.reli.(-k, x) > 0 && break
        x -= step
    end
    return x
end

function max_monotony(G::AMHGenerator)
    G.θ >= 0 && return Inf
    @inbounds for k in 3:100
        if G.θ < _find_critical_value_amh(k, step=1e-7)
            return k - 1
        end
    end
    return 100
end


ϕ(  G::AMHGenerator, t) = (1-G.θ)/(exp(t)-G.θ)
ϕ⁻¹(G::AMHGenerator, t) = log(G.θ + (1-G.θ)/t)
ϕ⁽¹⁾(G::AMHGenerator, t) = -((1-G.θ) * exp(t)) / (exp(t) - G.θ)^2
ϕ⁽ᵏ⁾(G::AMHGenerator, ::Val{k}, t) where k = (-1)^k * (1 - G.θ) / G.θ * PolyLog.reli(-k, G.θ * exp(-t))
ϕ⁻¹⁽¹⁾(G::AMHGenerator, t) = (G.θ - 1) / (G.θ * (t - 1) * t + t)
williamson_dist(G::AMHGenerator, ::Val{d}) where d = G.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-G.θ),Val{d}()) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),Val{d}())
frailty(G::AMHGenerator) = G.θ >= 0 ? Distributions.Geometric(1-G.θ) : throw("No frailty exists for AMH when θ < 0")
function _amh_tau(θ)
    if abs(θ) < 0.01
        return 2/9  * θ
            + 1/18  * θ^2
            + 1/45  * θ^3
            + 1/90  * θ^4
            + 2/315 * θ^5
            + 1/252 * θ^6
            + 1/378 * θ^7
            + 1/540 * θ^8
            + 2/1485 * θ^9
            + 1/990 * θ^10
    end
    if iszero(θ)
        return zero(θ)
    end
    u = isone(θ) ? θ : θ + (1-θ)^2 * log1p(-θ)
    return 1 - (2/3)*u/θ^2
end
τ(G::AMHGenerator) = _amh_tau(G.θ)
function τ⁻¹(::Type{T},tau) where T<:AMHGenerator
    if tau == zero(tau)
        return tau
    elseif tau > 1/3
        @info "AMHCopula cannot handle κ > 1/3."
        return one(tau)
    elseif tau < (5 - 8*log(2))/3
        @info "AMHCopula cannot handle κ < 5 - 8ln(2))/3 (approx -0.1817)."
        return -one(tau)
    end
    search_range = tau > 0 ? (0,1) : (-1,0)
    return Roots.find_zero(θ -> tau - _amh_tau(θ), search_range)
end

function _rho_amh(a)
    isnan(a) && return a
    aa = abs(a)
    aa < 7e-16 && return a / 3
    aa < 1e-4 && return (a / 3) * (1 + a / 4)
    aa < 0.002 && return a * (1/3 + a * (1/12 + a * (3/100)))
    aa < 0.007 && return a * (1/3 + a * (1/12 + a * (3/100 + a * (1/75))))
    aa < 0.016 && return a * (1/3 + a * (1/12 + a * (3/100 + a * (1/75 + a * (1/147)))))
    Li2 = PolyLog.reli2(a)  # dilog(a) = Li2(a)
    logTerm = (a < 1) ? 8 * (1 / a - 1) * log1p(-a) : 0.0
    return (3 / a) * (4 * (1 + 1 / a) * Li2 - logTerm - (a + 12))
end
ρ(G::AMHGenerator) = _rho_amh(G.θ)
function ρ⁻¹(::Type{AMHGenerator}, ρ::Real)
    ρ ≤ 33-48*log(2) && return -one(ρ)
    ρ ≥ 4pi^2 - 39 && return one(ρ)
    return Roots.find_zero(θ -> _rho_amh(θ) - ρ, (-1, 1), Roots.Brent())
end