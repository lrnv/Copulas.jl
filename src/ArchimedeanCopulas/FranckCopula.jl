"""
    FranckCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    FranckCopula(d, θ)

The [Franck](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [0,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = -\\frac{\\log\\left(1+e^{-t}(e^{-\\theta-1})\\right)}{\theta}
```
"""
struct FranckCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
FranckCopula(d,θ) = θ >= 0 ? FranckCopula{d,typeof(θ)}(θ) : @error "Theta must be positive"
ϕ(  C::FranckCopula,       t) = -log(1+exp(-t)*(exp(-C.θ)-1))/C.θ
ϕ⁻¹(C::FranckCopula,       t) = -log((exp(-t*C.θ)-1)/(exp(-C.θ)-1))

D₁ = GSL.sf_debye_1 # sadly, this is C code... corresponds to x -> x^{-1} * \int_0^x (t/(e^t-1)) dt

τ(C::FranckCopula) = 1+4(D₁(C.θ)-1)/C.θ

radial_dist(C::FranckCopula) = Logarithmic(1-exp(-C.θ))


