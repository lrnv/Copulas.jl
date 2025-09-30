"""
    AsymMixedTail{T}

Fields:
  - θ₁::Real — parameter
  - θ₂::Real — parameter

Constructor

  AsymMixedCopula((θ₁, θ₂))
  ExtremeValueCopula(2, AsymMixedTail((θ₁, θ₂)))

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
AsymMixedCopula(d::Integer, θ) = ExtremeValueCopula(2, AsymMixedTail(θ))
AsymMixedCopula(d::Integer, θ₁, θ₂) = AsymMixedCopula(d, (θ₁, θ₂))
Distributions.params(tail::AsymMixedTail) = (θ₁ = tail.θ₁, θ₂ = tail.θ₂)

function A(tail::AsymMixedTail, t::Real)
  θ₁, θ₂ = tail.θ₁, tail.θ₂
  tt = _safett(t)
  return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

# Fitting helpers for EV copulas using Asymmetric Mixed tail
_example(::Type{<:AsymMixedCopula}, d) = ExtremeValueCopula(2, AsymMixedTail((0.3, 0.2)))
_example(::Type{ExtremeValueCopula{2, AsymMixedTail}}, d) = ExtremeValueCopula(2, AsymMixedTail((0.3, 0.2)))
# Constraint set: θ₁ ≥ 0, θ₁ + θ₂ ≤ 1, θ₁ + 2θ₂ ≤ 1, θ₁ + 3θ₂ ≥ 0.
# We map α∈ℝ² → feasible (θ₁,θ₂) by using unconstrained (a,b) then projecting into a simple parameterization:
# Let s = σ(a) ∈ (0,1), t = σ(b) ∈ (0,1). Set θ₂ = t * min( (1 - s)/2, s/3 ) and θ₁ = s - θ₂.
_unbound_params(::Type{<:AsymMixedCopula}, d, θ) = begin
  # Inverse mapping is not unique; provide a smooth heuristic to get back to ℝ²
  # Recover s ≈ θ₁ + θ₂, t ≈ θ₂ / min((1-s)/2, s/3)
  s = clamp(θ.θ₁ + θ.θ₂, eps(), 1 - eps())
  m = min((1 - s)/2, s/3)
  t = m > 0 ? clamp(θ.θ₂ / m, eps(), 1 - eps()) : 0.5
  [log(s) - log1p(-s), log(t) - log1p(-t)]
end
_rebound_params(::Type{<:AsymMixedCopula}, d, α) = begin
  σ(x) = 1 / (1 + exp(-x))
  s = σ(α[1])
  t = σ(α[2])
  m = min((1 - s)/2, s/3)
  θ₂ = t * m
  θ₁ = s - θ₂
  (; θ₁, θ₂)
end
