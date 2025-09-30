"""
    InvGaussianGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    InvGaussianGenerator(θ)
    InvGaussianCopula(d,θ)

The Inverse Gaussian copula in dimension ``d`` is parameterized by ``\\theta \\in [0,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = \\exp\\left( \\frac{1 - \\sqrt{1 + 2\\theta^{2} t}}{\\theta} \\right).
```

More details about Inverse Gaussian Archimedean copula are found in :

    Mai, Jan-Frederik, and Matthias Scherer. Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 6. # N/A, 2017. Page 74.

Special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct InvGaussianGenerator{T} <: UnivariateFrailtyGenerator
    θ::T
    function InvGaussianGenerator(θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative."))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const InvGaussianCopula{d, T}   = ArchimedeanCopula{d, InvGaussianGenerator{T}}
InvGaussianCopula(d, θ)   = ArchimedeanCopula(d, InvGaussianGenerator(θ))
Distributions.params(G::InvGaussianGenerator) = (G.θ,)

ϕ(  G::InvGaussianGenerator, t) = isinf(G.θ) ? exp(-sqrt(2*t)) : exp((1-sqrt(1+2*((G.θ)^(2))*t))/G.θ)
ϕ⁻¹(G::InvGaussianGenerator, t) = isinf(G.θ) ? ln(t)^2/2 : ((1-G.θ*log(t))^(2)-1)/(2*(G.θ)^(2))
# ϕ⁽¹⁾(G::InvGaussianGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::InvGaussianGenerator, ::Val{k}, t) where k = kth derivative of ϕ
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
frailty(G::InvGaussianGenerator) = Distributions.InverseGaussian(G.θ,1)