"""
    AsymMixedTail{T}, AsymMixedCopula{T}

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


# Strictly invertible mapping from R^2 to the interior of the AsymMixedTail feasible set
# The feasible set is a convex quadrilateral with vertices:
# V1: (0, 0)
# V2: (1, 0)
# V3: (0, 1/3)
# V4: (1/2, 1/4)

function _rebound_params(::Type{<:AsymMixedTail}, d, α)
  # Map R^2 to (0,1)^2
  σ(x) = 1 / (1 + exp(-x))
  u, v = σ(α[1]), σ(α[2])
  # Map (u,v) to barycentric coordinates in the interior of the quadrilateral
  # Use (1-u)*(1-v), u*(1-v), (1-u)*v, u*v as weights for V1, V2, V3, V4
  w1 = (1-u)*(1-v)
  w2 = u*(1-v)
  w3 = (1-u)*v
  w4 = u*v
  # Vertices
  V1 = (0.0, 0.0)
  V2 = (1.0, 0.0)
  V3 = (0.0, 1/3)
  V4 = (1/2, 1/4)
  θ₁ = w1*V1[1] + w2*V2[1] + w3*V3[1] + w4*V4[1]
  θ₂ = w1*V1[2] + w2*V2[2] + w3*V3[2] + w4*V4[2]
  return (; θ₁, θ₂)
end

function _unbound_params(::Type{<:AsymMixedTail}, d, θ)
  # Inverse of the above: given (θ₁, θ₂) in the interior, recover (u,v) in (0,1)^2, then α
  # This is a nonlinear system, but for a convex quad, we can solve for (u,v) numerically
  # Use Newton's method or a simple fixed-point iteration
  function bary_inverse(θ₁, θ₂)
    # Vertices
    V1 = (0.0, 0.0)
    V2 = (1.0, 0.0)
    V3 = (0.0, 1/3)
    V4 = (1/2, 1/4)
    # Initial guess: project to (0,1)
    u, v = clamp(θ₁, 1e-6, 1-1e-6), clamp(θ₂*3, 1e-6, 1-1e-6)
    for _ in 1:20
      w1 = (1-u)*(1-v)
      w2 = u*(1-v)
      w3 = (1-u)*v
      w4 = u*v
      θ₁p = w1*V1[1] + w2*V2[1] + w3*V3[1] + w4*V4[1]
      θ₂p = w1*V1[2] + w2*V2[2] + w3*V3[2] + w4*V4[2]
      # Compute Jacobian
      dθ₁_du = (1-v)*(V2[1]-V1[1]) + v*(V4[1]-V3[1])
      dθ₁_dv = (1-u)*(V3[1]-V1[1]) + u*(V4[1]-V2[1])
      dθ₂_du = (1-v)*(V2[2]-V1[2]) + v*(V4[2]-V3[2])
      dθ₂_dv = (1-u)*(V3[2]-V1[2]) + u*(V4[2]-V2[2])
      # Newton step
      J = dθ₁_du*dθ₂_dv - dθ₁_dv*dθ₂_du
      if abs(J) < 1e-12; break; end
      du = ( (θ₁-θ₁p)*dθ₂_dv - (θ₂-θ₂p)*dθ₁_dv ) / J
      dv = ( (θ₂-θ₂p)*dθ₁_du - (θ₁-θ₁p)*dθ₂_du ) / J
      u = clamp(u + du, 1e-8, 1-1e-8)
      v = clamp(v + dv, 1e-8, 1-1e-8)
      if abs(du) < 1e-10 && abs(dv) < 1e-10; break; end
    end
    return u, v
  end
  u, v = bary_inverse(θ.θ₁, θ.θ₂)
  # Inverse sigmoid
  α1 = log(u/(1-u))
  α2 = log(v/(1-v))
  return [α1, α2]
end
function A(tail::AsymMixedTail, t::Real)
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    tt = _safett(t)
    return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

