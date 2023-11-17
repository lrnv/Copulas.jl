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

It has a few special cases: 
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
"""
struct GumbelCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function GumbelCopula(d,θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than or equal to 1"))
        elseif θ == 1
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::GumbelCopula,       t) = exp(-t^(1/C.θ))
ϕ⁻¹(C::GumbelCopula,       t) = (-log(t))^C.θ
τ(C::GumbelCopula) = ifelse(isfinite(C.θ), (C.θ-1)/C.θ, 1)
function τ⁻¹(::Type{GumbelCopula},τ) 
    if τ == 1
        return Inf
    else
        θ = 1/(1-τ)
        if θ < 1
            @warn "GumbelCopula cannot handle negative kendall tau's, returning independence.."
            return 1
        end
        return θ
    end
end
williamson_dist(C::GumbelCopula{d,T}) where {d,T} = WilliamsonFromFrailty(AlphaStable(α = 1/C.θ, β = 1,scale = cos(π/(2C.θ))^C.θ, location = (C.θ == 1 ? 1 : 0)), d)


# S(α, β, γ , δ) denotes a stable distribution in
# 1-parametrization [16, p. 8] with characteristic exponent α ∈ (0, 2], skewness β ∈ [−1, 1], scale
# γ ∈ [0,∞), and location δ ∈ R