"""
    AMHGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    AMHGenerator(θ)
    AMHCopula(d,θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t}-\\theta}
```

It has a few special cases: 
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct AMHGenerator{T} <: UnivariateGenerator
    θ::T
    function AMHGenerator(θ)
        if (θ < -1) || (θ > 1)
            throw(ArgumentError("Theta must be in [-1,1), you provided $θ."))
        elseif θ == 0
            return IndependentGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(::AMHGenerator) = Inf
ϕ(  G::AMHGenerator, t) = (1-G.θ)/(exp(t)-G.θ)
ϕ⁻¹(G::AMHGenerator, t) = log(G.θ + (1-G.θ)/t)
# ϕ⁽¹⁾(G::AMHGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::AMHGenerator, k, t) = kth derivative of ϕ
williamson_dist(G::AMHGenerator, d) = G.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-G.θ),d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)

function τ(G::AMHGenerator)
    θ = G.θ
    # unstable around zero, we instead cut its taylor expansion: 
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
    return Roots.find_zero(θ -> tau - τ(AMHGenerator(θ)), search_range)
end
