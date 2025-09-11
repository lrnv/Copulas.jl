"""
    AsymMixedTail{T}

Fields:
  - (θ₁, θ₂)::NTuple{2,Real}  — parameters

Constructor

    AsymMixedCopula(θ::AbstractVector)
    ExtremeValueCopula(AsymMixedTail(θ))

The (bivariate) asymmetric Mixed extreme-value copula is parameterized by two parameters ``\\theta_1``, ``\\theta_2`` subject to the following constraints:

* θ₁ ≥ 0
* θ₁ + θ₂ ≤ 1
* θ₁ + 2θ₂ ≤ 1
* θ₁ + 3θ₂ ≥ 0

Its Pickands dependence function is

```math
A(t) = \\theta_{2}t^3 + \\theta_{1}t^2 - (\\theta_1+\\theta_2)t + 1,\\quad t\\in[0,1].
```

Special cases:

* θ₁ = θ₂ = 0 ⇒ IndependentCopula
* θ₂ = 0      ⇒ symmetric Mixed copula

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct AsymMixedTail{T} <: Tail2
  θ₁::T
  θ₂::T
  function AsymMixedTail(θ)
      (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
      T = promote_type(eltype(θ))
      θ₁, θ₂ = (T(θ[1]), T(θ[2]))
      θ₁ == 0 && θ₂ == 0 && return NoTail()
      θ₂ == 0 && return MixedTail(θ₁)
      (θ₁ ≥ 0)             || throw(ArgumentError("θ₁ must be ≥ 0"))
      (θ₁ + θ₂ ≤ 1)        || throw(ArgumentError("θ₁+θ₂ ≤ 1"))
      (θ₁ + 2θ₂ ≤ 1)       || throw(ArgumentError("θ₁+2θ₂ ≤ 1"))
      (θ₁ + 3θ₂ ≥ 0)       || throw(ArgumentError("θ₁+3θ₂ ≥ 0"))
      new{T}(θ₁, θ₂)
  end
end

const AsymMixedCopula{T} = ExtremeValueCopula{2, AsymMixedTail{T}}
AsymMixedCopula(θ) = ExtremeValueCopula(2, AsymMixedTail(θ))
Distributions.params(tail::AsymMixedTail) = (tail.θ₁, tail.θ₂)

function A(tail::AsymMixedTail, t::Real)
  θ₁, θ₂ = tail.θ₁, tail.θ₂
  tt = _safett(t)
  return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end