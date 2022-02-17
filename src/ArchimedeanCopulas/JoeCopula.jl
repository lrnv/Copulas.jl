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
"""
struct JoeCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
JoeCopula(d,θ) = θ >= 1 ? JoeCopula{d,typeof(θ)}(θ) : @error "Theta must be greater than one."
ϕ(  C::JoeCopula,          t) = 1-(1-exp(-t))^(1/C.θ)
ϕ⁻¹(C::JoeCopula,          t) = -log(1-(1-t)^C.θ)
τ(C::JoeCopula) = 1 - 4sum(1/(k*(2+k*C.θ)*(C.θ*(k-1)+2)) for k in 1:1000) # 446 in R copula. 
radial_dist(C::JoeCopula) = Sibuya(1/C.θ)


