"""
    AsymMixedTail{T}

Fields:
  - θ::NTuple{2,Real}  — parameters

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
* θ₂ = 0       ⇒ symmetric Mixed copula

References:

* Tawn (1988). Bivariate extreme value theory: models and estimation. Biometrika 75(3): 397-415.
  """
  struct AsymMixedTail{T} <: Tail{2}
    θ::NTuple{2,T}
    function AsymMixedTail(θ)
        (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
        T = promote_type(eltype(θ))
        θ₁, θ₂ = (T(θ[1]), T(θ[2]))
        (θ₁ ≥ 0)             || throw(ArgumentError("θ₁ must be ≥ 0"))
        (θ₁ + θ₂ ≤ 1)        || throw(ArgumentError("θ₁+θ₂ ≤ 1"))
        (θ₁ + 2θ₂ ≤ 1)       || throw(ArgumentError("θ₁+2θ₂ ≤ 1"))
        (θ₁ + 3θ₂ ≥ 0)       || throw(ArgumentError("θ₁+3θ₂ ≥ 0"))
        new{T}((θ₁, θ₂))
    end
  end

const AsymMixedCopula{T} = ExtremeValueCopula{2, AsymMixedTail{T}}
Distributions.params(C::ExtremeValueCopula{2, AsymMixedTail{T}}) where {T} = (C.E.θ[1], C.E.θ[2])

function A(E::AsymMixedTail, t::Real)
  θ₁, θ₂ = E.θ
  tt = _safett(t)
  return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

function AsymMixedCopula(θ::AbstractVector)
  (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
  θt = (θ[1], θ[2])
  if θt[1] == 0 && θt[2] == 0
    return IndependentCopula(2)
  elseif θt[2] == 0
    return MixedCopula(θt[1])
  else
    return ExtremeValueCopula( AsymMixedTail(θt) )
  end
end

AsymMixedCopula(θ::NTuple{2,Any}) = AsymMixedCopula(collect(θ))