"""
    InvGaussianGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    InvGaussianGenerator(θ)
    InvGaussianCopula(d,θ)

The Inverse Gaussian copula in dimension ``d`` is parameterized by ``\\theta \\in [0,\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp{\\frac{1-\\sqrt{1+2θ^{2}t}}{θ}}.
```

More details about Inverse Gaussian Archimedean copula are found in :

    Mai, Jan-Frederik, and Matthias Scherer. Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 6. # N/A, 2017. Page 74.

It has a few special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct InvGaussianGenerator{T} <: Generator
    θ::T
    function InvGaussianGenerator(θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative."))
        elseif θ == 0
            return IndependentGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
const InvGaussianCopula{d, T}   = ArchimedeanCopula{d, InvGaussianGenerator{T}}
InvGaussianCopula(d, θ)   = ArchimedeanCopula(d, InvGaussianGenerator(θ))

Distributions.params(C::InvGaussianCopula) = (C.G.θ)

max_monotony(G::InvGaussianGenerator) = Inf
ϕ(  G::InvGaussianGenerator, x) = isinf(G.θ) ? exp(-sqrt(2*x)) : exp((1 - sqrt(1 + 2*(G.θ^2)*x))/G.θ)
ϕ⁻¹(G::InvGaussianGenerator, u) = isinf(G.θ) ? (log(u)^2)/2 : ((1 - G.θ*log(u))^2 - 1) / (2*(G.θ^2))
function ϕ⁽¹⁾(G::InvGaussianGenerator, x)
    if isinf(G.θ)
        # d/dx e^{-√(2x)} = e^{-√(2x)} * ( -1 / √(2x) )
        return ϕ(G, x) * ( - 1 / sqrt(2*x) )
    else
        r = sqrt(1 + 2*(G.θ^2)*x)
        return ϕ(G, x) * ( - G.θ / r )
    end
end
ϕ⁻¹⁽¹⁾(G::InvGaussianGenerator, u::Real) = isinf(G.θ) ? log(u) / u : (G.θ*log(u) - 1) / (G.θ*u)

function _invgaussian_tau(θ)
    T = promote_type(typeof(θ),Float64)
    if θ == 0
        return zero(θ)
    elseif θ > 1e153 # should be Inf, but integrand has issues...
        return 1/2
    elseif θ < sqrt(eps(T))
        return zero(θ)
    end
    function _integrand(x,θ)
        y = 1-θ*log(x)
        ret = - x*(y^2-1)/(2θ*y)
        return ret
    end
    rez, err = QuadGK.quadgk(x -> _integrand(x,θ),zero(θ),one(θ))
    rez = 1+4*rez
    return rez
end
τ(G::InvGaussianGenerator) = _invgaussian_tau(G.θ)
function τ⁻¹(::Type{T}, tau) where T<:InvGaussianGenerator
    if tau == zero(tau)
        return tau
    elseif tau < 0
        @info "InvGaussianCopula cannot handle κ < 0."
        return zero(tau)
    elseif tau > 0.5
        @info "InvGaussianCopula cannot handle κ > 1/2."
        return tau * Inf
    end
    return Roots.find_zero(x -> _invgaussian_tau(x) - tau, (sqrt(eps(tau)), Inf))
end
williamson_dist(G::InvGaussianGenerator, ::Val{d}) where d = WilliamsonFromFrailty(Distributions.InverseGaussian(G.θ,1), Val{d}())
frailty_dist(G::InvGaussianGenerator) = Distributions.InverseGaussian(G.θ,1)