"""
    JoeCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    JoeCopula(d, θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = 1 - \\left(1 - e^{-t}\\right)^{\\frac{1}{\\theta}}
```

It has a few special cases: 
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
"""
struct JoeCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function JoeCopula(d,θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::JoeCopula,          t) = 1-(-expm1(-t))^(1/C.θ)
ϕ⁻¹(C::JoeCopula,          t) = -log1p(-(1-t)^C.θ)
_joe_tau_f(θ) = 1 - 4sum(1/(k*(2+k*θ)*(θ*(k-1)+2)) for k in 1:1000) # 446 in R copula. 
τ(C::JoeCopula) = _joe_tau_f(C.θ)
function τ⁻¹(::Type{JoeCopula},τ) 
    if τ == 1
        return Inf
    elseif τ == 0
        return 1
    elseif τ < 0
        @warn "JoeCoula cannot handle negative kendall taus, we return the independence..."
        return one(τ)
    else
        return Roots.find_zero(θ -> _joe_tau_f(θ) - τ, (one(τ),τ*Inf))
    end
end
williamson_dist(C::JoeCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Sibuya(1/C.θ), d)


