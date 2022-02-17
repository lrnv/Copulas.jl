"""
    GumbelCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelCopula(d, θ)

The [Gumbel](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = \\exp{-t^{\\frac{1}{θ}}}
```
"""
struct GumbelCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
GumbelCopula(d,θ) = θ >= 1 ? GumbelCopula{d,typeof(θ)}(θ) : @error "Theta must be greater than 1."
ϕ(  C::GumbelCopula,       t) = exp(-t^(1/C.θ))
ϕ⁻¹(C::GumbelCopula,       t) = (-log(t))^C.θ
τ(C::GumbelCopula) = (C.θ-1)/C.θ
τ⁻¹(::Type{GumbelCopula},τ) =1/(1-τ) 

function radial_dist(C::GumbelCopula)
    α = 1/C.θ
    β = 1
    γ = cos(π/(2C.θ))^C.θ
    δ = C.θ == 1 ? 1 : 0
    AlphaStable([α,β,γ,δ]...) # for the type promotion...
end


# S(α, β, γ , δ) denotes a stable distribution in
# 1-parametrization [16, p. 8] with characteristic exponent α ∈ (0, 2], skewness β ∈ [−1, 1], scale
# γ ∈ [0,∞), and location δ ∈ R