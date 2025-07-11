"""
    GumbelBarnettGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelBarnettGenerator(θ)
    GumbelBarnettCopula(d,θ)

The Gumbel-Barnett copula is an archimdean copula with generator:

```math
\\phi(t) = \\exp{θ^{-1}(1-e^{t})}, 0 \\leq \\theta \\leq 1.
```

It has a few special cases:
- When θ = 0, it is the IndependentCopula

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.437
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GumbelBarnettGenerator{T} <: UnivariateGenerator
    θ::T
    function GumbelBarnettGenerator(θ)
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::GumbelBarnettGenerator) = Inf
# generator
ϕ(G::GumbelBarnettGenerator, t::Number) = exp((1 - exp(t)) / G.θ)
# first generator derivative
ϕ⁽¹⁾(G::GumbelBarnettGenerator, t::Real) = -exp(1 - exp(t) / (G.θ + t)) / G.θ
# inverse generator
ϕ⁻¹(G::GumbelBarnettGenerator, t::Real) = log1p(-G.θ * log(t))
# first inverse generator derivative
ϕ⁻¹⁽¹⁾(G::GumbelBarnettGenerator, t::Real) = -G.θ / (t - G.θ * t * log(t))

function τ(G::GumbelBarnettGenerator)
    # Use a numerical integration method to obtain tau
    r, _ = QuadGK.quadgk(x -> (1 - G.θ * log(x)) * log1p(-G.θ * log(x)) * x, 0, 1)
    return 1 - 4 * r / G.θ
end
function τ⁻¹(::Type{T}, tau) where {T<:GumbelBarnettGenerator}
    if tau == 0
        return zero(tau)
    elseif tau > 0
        @info "GumbelBarnettCopula cannot handle κ > 0."
        return zero(tau)
    elseif tau < τ(GumbelBarnettCopula(2, 1))
        @info "GumbelBarnettCopula cannot handle κ <≈ -0.3613."
        return one(tau)
    end
    # Use the bisection method to find the root
    x = Roots.find_zero(x -> τ(GumbelBarnettGenerator(x)) - tau, (0.0, 1.0))
    return x
end
