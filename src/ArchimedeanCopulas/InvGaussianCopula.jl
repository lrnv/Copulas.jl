"""
    InvGaussianCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    InvGaussianCopula(d, θ)

The Inverse Gaussian copula in dimension ``d`` is parameterized by ``\\theta \\in [0,\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp{\\frac{(1-\\sqrt{1+2θ^{2}t}}{θ}}.
```

More details about Inverse Gaussian Archimedean copula are found in :

    Mai, Jan-Frederik, and Matthias Scherer. Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 6. # N/A, 2017. Page 74.

It has a few special cases:
- When θ = 0, it is the IndependentCopula
"""
struct InvGaussianCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function InvGaussianCopula(d,θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative."))
        elseif θ == 0
            return IndependentCopula(d)
        # elseif isinf(θ)
        #     throw(ArgumentError("Theta cannot be infinite"))
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end

ϕ(  C::InvGaussianCopula, t) = isinf(C.θ) ? exp(-sqrt(2*t)) : exp((1-sqrt(1+2*((C.θ)^(2))*t))/C.θ)
ϕ⁻¹(C::InvGaussianCopula, t) = isinf(C.θ) ? ln(t)^2/2 : ((1-C.θ*log(t))^(2)-1)/(2*(C.θ)^(2))
function _invg_tau_f(θ)
    if θ == 0
        return zero(θ)
    elseif θ > 1e153 # should be Inf, but integrand has issues... 
        return 1/2
    elseif θ < sqrt(eps(θ))
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
τ(C::InvGaussianCopula) = _invg_tau_f(C.θ)
function τ⁻¹(::Type{InvGaussianCopula}, tau)
    if tau == zero(tau)
        return tau
    elseif tau < 0
        @warn "InvGaussianCopula cannot handle negative dependencies, returning independence..."
        return zero(tau)
    elseif tau > 0.5
        @warn "InvGaussianCopula cannot handle kendall tau greater than 0.5, using 0.5.."
        return tau * Inf
    end
    return Roots.find_zero(x -> _invg_tau_f(x) - tau, (sqrt(eps(tau)), Inf))
end

williamson_dist(C::InvGaussianCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Distributions.InverseGaussian(C.θ,1),d)