"""
    ClaytonCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = \\left(1+\\mathrm{sign}(\\theta)*t\\right)^{-1\\frac{1}{\\theta}}
```

It has a few special cases: 
- When θ = -1/(d-1), it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 0, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
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
ϕ(  C::ClaytonCopula,      t) = max(1+C.θ*t,zero(t))^(-1/C.θ)
ϕ⁻¹(C::ClaytonCopula,      t) = (t^(-C.θ)-1)/C.θ
τ(C::ClaytonCopula) = ifelse(isfinite(C.θ), C.θ/(C.θ+2), 1)
τ⁻¹(::Type{ClaytonCopula},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))
williamson_dist(C::ClaytonCopula{d,T}) where {d,T} = C.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,1),d) : ClaytonWilliamsonDistribution(C.θ,d)