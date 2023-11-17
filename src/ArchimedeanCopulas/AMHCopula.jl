"""
    AMHCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    AMHCopula(d, θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t}-\\theta}
```

It has a few special cases: 
- When θ = 0, it is the IndependentCopula
"""
const AMHCopula{d,T} = ArchimedeanCopula{d,AMHGenerator{T}}
AMHCopula(d,θ) = ArchimedeanCopula(d,AMHGenerator(θ))

_amh_tau_f(θ) = θ == 1 ? 1/3 : 1 - 2(θ+(1-θ)^2*log1p(-θ))/(3θ^2)
τ(C::AMHCopula) = _amh_tau_f(C.θ) # no closed form inverse...
function τ⁻¹(::Type{AMHCopula},τ)
    if τ == zero(τ)
        return τ
    end
    if τ > 1/3
        @warn "AMHCopula cannot handle kendall tau's greater than 1/3. We capped it to 1/3."
        return one(τ)
    end
    if τ < (5 - 8*log(2))/3
        @warn "AMHCopula cannot handle kendall tau's smaller than (5- 8ln(2))/3 (approx -0.1817). We capped it to this value."
        return -one(τ)
    end
    return Roots.find_zero(θ -> _amh_tau_f(θ) - τ, (-one(τ), one(τ)))
end