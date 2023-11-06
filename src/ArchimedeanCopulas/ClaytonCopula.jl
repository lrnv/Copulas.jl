"""
    ClaytonCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,\\infty)`` when ``d=2`` and ``\\theta \\in [0,\\infty)`` if ``d>2``. It is an Archimedean copula with generator : 

```math
\\phi(t) = \\left(1+\\mathrm{sign}(\\theta)*t\\right)^{-1\\frac{1}{\\theta}}
```
"""
struct ClaytonCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function ClaytonCopula(d,θ)
        if θ < -1/(d-1)
            throw(ArgumentError("Theta must be greater than -1/(d-1)"))
        elseif θ == -1/(d-1)
            return WCopula(d)
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::ClaytonCopula,      t) = (1+sign(C.θ)*t)^(-1/C.θ)
ϕ⁻¹(C::ClaytonCopula,      t) = sign(C.θ)*(t^(-C.θ)-1)
radial_dist(C::ClaytonCopula) = Distributions.Gamma(1/C.θ,1) # Currently fails for negative thetas ! thus negtatively correlated clayton copulas cannot be sampled...
τ(C::ClaytonCopula) = ifelse(isfinite(C.θ), C.θ/(C.θ+2), 1)
τ⁻¹(::Type{ClaytonCopula},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))