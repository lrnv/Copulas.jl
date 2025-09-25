"""
    AsymMixedTail{T}

Fields:
  - θ₁::Real — parameter
  - θ₂::Real — parameter

Constructor

  AsymMixedCopula(θ₁, θ₂)
  ExtremeValueCopula(2, AsymMixedTail(θ₁, θ₂))

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
  function AsymMixedTail(θ₁, θ₂)
      θ₁, θ₂ = promote(θ₁, θ₂)
      T = typeof(θ₁)
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
Distributions.params(tail::AsymMixedTail) = (θ₁ = tail.θ₁, θ₂ = tail.θ₂)
_unbound_params(::Type{<:AsymMixedTail}, d, θ) = begin
  # Inverse mapping is not unique; provide a smooth heuristic to get back to ℝ²
  # Recover s ≈ θ₁ + θ₂, t ≈ θ₂ / min((1-s)/2, s/3)
  s = clamp(θ.θ₁ + θ.θ₂, eps(), 1 - eps())
  m = min((1 - s)/2, s/3)
  t = m > 0 ? clamp(θ.θ₂ / m, eps(), 1 - eps()) : 0.5
  [log(s) - log1p(-s), log(t) - log1p(-t)]
end
_rebound_params(::Type{<:AsymMixedTail}, d, α) = begin
  σ(x) = 1 / (1 + exp(-x))
  s = σ(α[1])
  t = σ(α[2])
  m = min((1 - s)/2, s/3)
  θ₂ = t * m
  θ₁ = s - θ₂
  (; θ₁, θ₂)
end

function A(tail::AsymMixedTail, t::Real)
  θ₁, θ₂ = tail.θ₁, tail.θ₂
  tt = _safett(t)
  return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

