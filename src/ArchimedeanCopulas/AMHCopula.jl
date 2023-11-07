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
struct AMHCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function AMHCopula(d,θ)
        if (θ < -1) || (θ >= 1)
            throw(ArgumentError("Theta must be in [-1,1)"))
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ < 0
            return WilliamsonCopula(t -> (1-θ)/(exp(t)-θ),d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::AMHCopula,t) = (1-C.θ)/(exp(t)-C.θ)
ϕ⁻¹(  C::AMHCopula,t) = log(C.θ + (1-C.θ)/t)

τ(C::AMHCopula) = 1 - 2(C.θ+(1-C.θ)^2*log(1-C.θ))/(3C.θ^2) # no closed form inverse...
function τ⁻¹(::Type{AMHCopula},τ)
    if τ == zero(τ)
        return τ
    end
    if τ > 1/3
        @warn "AMHCopula cannot handle kendall tau's greater than 1/3. We capped it to 1/3."
        return 1
    end
    return Roots.fzero(θ -> 1 - 2(θ+(1-θ)^2*log(1-θ))/(3θ^2) - τ,0.5)
end


frailty_dist(C::AMHCopula) = 1 + Distributions.Geometric(1-C.θ)


