"""
    SibuyaStoppedPosStable(θ, δ)

Parameters
    * `θ ≥ 1`
    * `δ ≥ 1`

Used as frailty for Archimedean copulas (Sibuya‑positive‑stable mixture).

Hierarchical frailty:
```math
T \\sim \\mathrm{Sibuya}(1/θ), \\qquad X \\mid T=t \\sim \\mathrm{PositiveStable}(1/δ; \\text{scale}= t^{δ}).
```
θ ≥ 1, δ ≥ 1.
"""
struct SibuyaStoppedPosStable{T} <: Distributions.ContinuousUnivariateDistribution
    θ::T   # ≥ 1
    δ::T   # ≥ 1
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::SibuyaStoppedPosStable)
    θ = float(D.θ); δ = float(D.δ)
    t = rand(rng, Sibuya(1/θ))                 # T ~ Sibuya(1/θ)
    return rand(rng, PStable(1/δ; scale = t^δ))# M | T=t ~ Postable(α=1/δ, scale=t^δ)
end
