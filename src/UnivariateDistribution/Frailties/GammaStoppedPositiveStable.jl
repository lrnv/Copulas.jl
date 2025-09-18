"""
    GammaStoppedPositiveStable(α, β)

Parameters
    * `α ∈ (0,1]`
    * `β > 0`

Used as frailty for Archimedean copulas (gamma mixture of positive stable).

Hierarchical positive frailty:
```math
T \\sim \\mathrm{Gamma}(β,1), \\qquad X \\mid T=t \\sim \\mathrm{PositiveStable}(α; \\text{scale}= t^{1/α}).
```
Parameters: α ∈ (0,1], β > 0. Used to derive Archimedean generators via `E[e^{-sX}]`.

"""
struct GammaStoppedPositiveStable{Tα,Tβ} <: Distributions.ContinuousUnivariateDistribution
    α::Tα   # = 1/δ
    β::Tβ   # = 1/θ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedPositiveStable)
    α, β = float(D.α), float(D.β)
    T = rand(rng, Distributions.Gamma(β, 1.0))        # Gamma(shape=β, scale=1)
    return rand(rng, PStable(α; scale = T^(1/α)))     # Z(T)
end
