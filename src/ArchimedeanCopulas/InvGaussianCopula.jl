"""
InvGaussianCopulaCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    InvGaussianCopula(d, θ)

The Inverse Gaussian copula is an archimdean copula with generator:

```math
\\phi(t) = \\exp{(1-\\sqrt{1+2θ^{2}t})/θ}, \\theta > 0.
```

More details about Inverse Gaussian Archimedean copula are found in :
    Mai, Jan-Frederik, and Matthias Scherer. 
    Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 6. # N/A, 2017. Page 74.
"""

struct InvGaussianCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function InvGaussianCopula(d,θ)
        if θ < 0
            throw(ArgumentError("Theta must be greater than 0"))
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end

ϕ(  C::InvGaussianCopula,       t) = exp((1-sqrt(1+2*((C.θ)^(2))*t))/C.θ)
ϕ⁻¹(C::InvGaussianCopula,       t) = ((1-C.θ*log(t))^(2)-1)/(2*(C.θ)^(2))
function τ(C::InvGaussianCopula)
    # Define the function to integrate
    f(x) = -x * (((1 - C.θ * log(x))^2 - 1) / (2 * C.θ * (1 - C.θ * log(x))))

    # Calculate the integral using an appropriate numerical integration method
    result, _ = gsl_integration_qags(f, 0.0, 1.0, [C.θ], 1e-7, 1000)

    return 1+4*result
end
function τ⁻¹(::Type{InvGaussianCopula}, τ)
    if τ == zero(τ)
        return τ
    end

    # Define an anonymous function that takes a value x and computes τ for an InvGaussianCopula copula with θ = x
    τ_func(x) = τ(InvGaussianCopula{d, Float64}(x))

    # Set an initial value for x₀ (adjustable)
    x₀ = (1-τ)/4

    return Roots.find_zero(x -> τ_func(x) - τ, x₀)
end

williamson_dist(C::InvGaussianCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Distributions.InverseGaussian(C.θ,1),d)