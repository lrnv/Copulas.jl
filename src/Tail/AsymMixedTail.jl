"""
    AsymMixedTail{T}, AsymMixedCopula{T}

Fields:
  - θ₁::Real — parameter
  - θ₂::Real — parameter

Constructor

  AsymMixedCopula(2, θ₁, θ₂)
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
AsymMixedTail, AsymMixedCopula

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


# The generic ExtremeValueCopula `_example` uses equal unconstrained
# coordinates. Under the AsymMixed map that gives u=v and therefore θ₂=0,
# which intentionally simplifies to MixedTail.  Fitting must instead start
# from a genuinely asymmetric interior point so that params(_example(...))
# keeps the (θ₁, θ₂) interface.
function _example(CT::Type{<:ExtremeValueCopula{2,<:AsymMixedTail}}, d::Int,)
    d == 2 || throw(DimensionMismatch("AsymMixedCopula is only defined in dimension two",))
    return CT(d, 0.50, 0.10)
end


# Strictly invertible mapping from R^2 to the interior of the actual
# AsymMixedTail feasible set
#
#   θ₁ ≥ 0,
#   θ₁ + θ₂ ≤ 1,
#   θ₁ + 2θ₂ ≤ 1,
#   θ₁ + 3θ₂ ≥ 0.
#
# Its vertices are
#
#   (0, 0), (0, 1/2), (1, 0), (3/2, -1/2).
#
# Map the open unit square bilinearly to this quadrilateral using corners
# V00=(0,0), V10=(3/2,-1/2), V01=(0,1/2), V11=(1,0).
# For u,v in (0,1) this simplifies to
#
#   θ₁ = u(3-v)/2,
#   θ₂ = (v-u)/2.
function _rebound_params(::Type{<:AsymMixedTail}, d, α)
    σ(x) = inv(1 + exp(-x))
    u, v = σ(α[1]), σ(α[2])

    θ₁ = u * (3 - v) / 2
    θ₂ = (v - u) / 2
    return (; θ₁, θ₂)
end

function _unbound_params(::Type{<:AsymMixedTail}, d, θ)
    θ₁ = float(θ.θ₁)
    θ₂ = float(θ.θ₂)

    # Invert
    #   θ₂ = (v-u)/2,
    #   θ₁ = u(3-v)/2.
    # Substituting v=u+2θ₂ gives
    #
    #   u² - (3-2θ₂)u + 2θ₁ = 0.
    #
    # Use the stable expression for the smaller root, which is the one
    # lying in [0,1] on the feasible quadrilateral.
    b = 3 - 2θ₂
    disc = max(b*b - 8θ₁, 0.0)
    root = sqrt(disc)

    u = iszero(θ₁) ? 0.0 : (4θ₁) / (b + root)
    v = u + 2θ₂

    # _unbound_params is used by unconstrained fitting; finite values at
    # feasible boundaries are preferable to ±Inf.
    δ = sqrt(eps(Float64))
    u = clamp(u, δ, 1 - δ)
    v = clamp(v, δ, 1 - δ)

    return [log(u) - log1p(-u), log(v) - log1p(-v)]
end

function A(tail::AsymMixedTail, t::Real)
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    tt = _safett(t)
    return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

function dA(tail::AsymMixedTail, t::Real)
    tt = _safett(t)
    θ₁, θ₂ = tail.θ₁, tail.θ₂

    return 3θ₂ * tt^2 + 2θ₁ * tt - (θ₁ + θ₂)
end

function d²A(tail::AsymMixedTail, t::Real)
    tt = _safett(t)
    θ₁, θ₂ = tail.θ₁, tail.θ₂

    return 6θ₂ * tt + 2θ₁
end