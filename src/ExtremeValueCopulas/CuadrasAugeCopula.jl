"""
    CuadrasAugeCopula{P}

Fields:

    - α::Real - parameter
    
Constructor

    CuadrasAugeCopula(α)

The bivariate Cuadras Auge copula is parameterized by ``\\alpha \\in [0,1]``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\max\\{t, 1-t \\} + (1-\\theta)\\max\\{t, 1-t\\}
```

References:
* Simulating copulas: stochastic models, sampling algorithms, and applications. 2017.
"""
struct CuadrasAugeCopula{P} <: ExtremeValueCopula{P}
    θ::P  # Copula parameter
    function CuadrasAugeCopula(θ)
        if !(0 <= θ <= 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentCopula(2)
        elseif θ == 1
            return MCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end

𝘈(C::CuadrasAugeCopula, t::Real) = max(t, 1-t) + (1-C.θ) * min(t, 1-t)

d𝘈(C::CuadrasAugeCopula, t::Real) = t <= 0.5 ? -C.θ : C.θ

τ(C::CuadrasAugeCopula) = C.θ/(2-C.θ)

ρₛ(C::CuadrasAugeCopula) = (3*C.θ)/(4-C.θ)

# specific ℓ específica of Cuadras-Augé Copula
function ℓ(C::CuadrasAugeCopula, t::Vector)
    θ = C.θ
    t₁, t₂ = t
    return max(t₁, t₂) + (1-θ) * min(t₁, t₂)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CuadrasAugeCopula, x::AbstractVector{T}) where {T<:Real}
    θ = C.θ
    E₁, E₂ = rand(rng, Distributions.Exponential(θ/(1-θ)),2)
    E₁₂ = rand(rng, Distributions.Exponential())
    x[1] = exp(-(1/θ)*min(E₁,E₁₂))
    x[2] = exp(-(1/θ)*min(E₂,E₁₂))
    return x
end