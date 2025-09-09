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

* Tawn (1988). Bivariate extreme value theory: models and estimation. Biometrika 75(3): 397-415.
"""
struct MixedTail{T} <: Tail{2}
θ::T
function MixedTail(θ)
(0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in \[0,1]"))
return new{typeof(θ)}(θ)
end
end

const MixedCopula{T} = ExtremeValueCopula{2, MixedTail{T}}
Distributions.params(C::ExtremeValueCopula{2,MixedTail{T}}) where {T} = (C.E.θ,)

function A(E::MixedTail, t::Real)
    tt = _safett(t)
    θ  = E.θ
    return θ*tt^2 - θ*tt + 1
end

function MixedCopula(θ)
    if θ == 0
        return IndependentCopula(2)
    else
        return ExtremeValueCopula(MixedTail(θ))
    end
end