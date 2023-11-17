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
        elseif isinf(θ)
            throw(ArgumentError("Theta cannot be infinite"))
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end

ϕ(  C::InvGaussianCopula,       t) = exp((1-sqrt(1+2*((C.θ)^(2))*t))/C.θ)
ϕ⁻¹(C::InvGaussianCopula,       t) = ((1-C.θ*log(t))^(2)-1)/(2*(C.θ)^(2))
function τ(C::InvGaussianCopula)

    # Calculate the integral using an appropriate numerical integration method
    result, _ = QuadGK.quadgk( x -> (x*((1-C.θ*log(x))^2-1))/(-2*C.θ*(1-C.θ*log(x))),0,1)

    return 1+4*result
end
function τ⁻¹(::Type{InvGaussianCopula}, tau)
    if tau == zero(tau)
        return tau
    end
    if tau < 0
        @warn "InvGaussianCopula cannot handle negative dependencies, returning independence..."
        return zero(τ)
    end

    # Define an anonymous function that takes a value x and computes τ for an InvGaussianCopula copula with θ = x
    τ_func(x) = τ(InvGaussianCopula(2,x))

    # Set an initial value for x₀ (adjustable)
    x = Roots.find_zero(x -> τ_func(x) - tau, (0.0, Inf))
    return τ
end

williamson_dist(C::InvGaussianCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Distributions.InverseGaussian(C.θ,1),d)