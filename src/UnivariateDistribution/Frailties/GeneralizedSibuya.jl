"""
    GeneralizedSibuya(ϑ, δ)

Parameters
    * `ϑ ≥ 1`
    * `δ ∈ (0,1]`

Used as flexible frailty (generalizing Sibuya) for Archimedean generators.

Generalization of Sibuya law with parameters ϑ ≥ 1 and δ ∈ (0,1]. Admits mixed
accept–reject constructions via either a logarithmic or Sibuya proposal depending
on δ. Appears as frailty to generate flexible Archimedean families.

Sampling algorithm switches regime at δ = 1 - exp(-1) for efficiency.
"""
struct GeneralizedSibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    ϑ::T
    δ::T
    function GeneralizedSibuya(ϑ, δ)
        (ϑ ≥ 1)  || throw(ArgumentError("ϑ must be ≥ 1"))
        (0 < δ ≤ 1) || throw(ArgumentError("δ must be in (0,1]"))
        new{typeof(ϑ)}(ϑ, δ)
    end
end

function Distributions.rand(rng::Distributions.AbstractRNG, d::GeneralizedSibuya)
    ϑ = float(d.ϑ); δ = float(d.δ)
    η = 1 - (1 - δ)^ϑ
    if δ ≤ 1 - exp(-1)
        logser = Logarithmic(log1p(-η))
        while true
            X = rand(rng, logser)
            acc = 1 / ((X - 1/ϑ) * SpecialFunctions.beta(X, 1 - 1/ϑ))
            if rand(rng, Distributions.Uniform(0,1)) ≤ acc
                return X
            end
        end
    else                   
        α = 1/ϑ            
        while true
            X = rand(rng, Sibuya(α))
            if rand(rng, Distributions.Uniform(0,1)) ≤ η^(X-1) 
                return X
            end
        end
    end
end