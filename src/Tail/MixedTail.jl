"""
    MixedTail{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    MixedCopula(θ)
    ExtremeValueCopula(MixedTail(θ))

The (bivariate) Mixed extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\theta t^2 - \\theta t + 1, \\quad t \\in [0,1].
```

Special cases:

* θ = 0 ⇒ IndependentCopula

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct MixedTail{T} <: Tail2
    θ::T
    function MixedTail(θ)
        (0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in \[0,1]"))
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const MixedCopula{T} = ExtremeValueCopula{2, MixedTail{T}}
Distributions.params(tail::MixedTail) = (tail.θ,)
A(E::MixedTail, t::Real) = E.θ * t^2 - E.θ * t + 1