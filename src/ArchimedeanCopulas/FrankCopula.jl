"""
    FrankCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    FrankCopula(d, θ)

The [Frank](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-\\infty,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = -\\frac{\\log\\left(1+e^{-t}(e^{-\\theta-1})\\right)}{\theta}
```

It has a few special cases: 
- When θ = -∞, it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

"""
struct FrankCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function FrankCopula(d,θ)
        if θ == -Inf
            return WCopula(d)
        elseif θ == 1
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::FrankCopula,       t) = -log(1+exp(-t)*(exp(-C.θ)-1))/C.θ
ϕ⁻¹(C::FrankCopula,       t) = -log((exp(-t*C.θ)-1)/(exp(-C.θ)-1))

D₁ = GSL.sf_debye_1 # sadly, this is C code.
# could be replaced by : 
# using QuadGK
# D₁(x) = quadgk(t -> t/(exp(t)-1), 0, x)[1]/x
# to make it more general. but once gain, it requires changing the integrator at each evlauation, 
# which is problematic. 
# Better option is to try to include this function into SpecialFunctions.jl. 


τ(C::FrankCopula) = 1+4(D₁(C.θ)-1)/C.θ
function τ⁻¹(::Type{FrankCopula},τ)
    if τ == zero(τ)
        return τ
    end
    x₀ = (1-τ)/4
    return Roots.fzero(x -> (1-D₁(x))/x - x₀, 1e-4, Inf)
end
    

radial_dist(C::FrankCopula) = Logarithmic(1-exp(-C.θ))


