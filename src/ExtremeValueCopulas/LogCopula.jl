"""
    LogCopula{P}

Fields:

    - θ::Real - parameter
    
Constructor

    LogCopula(θ)

The bivariate Logistic copula (or Gumbel Copula) is parameterized by ``\\theta \\in [1,\\infty)``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = (t^{\\theta}+(1-t)^{\\theta})^{\\frac{1}{\\theta}}
```

It has a few special cases: 
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct LogCopula{P} <: ExtremeValueCopula{P}
    θ::P  # Copula parameter
    function LogCopula(θ)
        if !(1 <= θ)
            throw(ArgumentError(" The param θ must be in [1, ∞)"))
        elseif θ == 1
            return IndependentCopula(2)
        elseif θ == Inf
            return MCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end
# #  specific ℓ funcion of LogCopula
function ℓ(G::LogCopula, t::Vector)
    θ = G.θ
    t₁, t₂ = t
    return (t₁^θ + t₂^θ)^(1/θ)
end
# #  specific A funcion of LogCopula
A(C::LogCopula, t::Real) = exp(LogExpFunctions.logaddexp(C.θ*log(t),C.θ*log(1-t))/C.θ)
