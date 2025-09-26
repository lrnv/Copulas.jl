"""
    SibuyaStoppedGamma(θ, δ)

Parameters
    * `θ ≥ 1`
    * `δ > 0`

Used as frailty for Archimedean copulas (Sibuya‑gamma mixture).

Hierarchical frailty:
```math
T \\sim \\mathrm{Sibuya}(1/θ), \\qquad X \\mid T=t \\sim \\mathrm{Gamma}(t/δ, 1).
```
θ ≥ 1, δ > 0.
"""
struct SibuyaStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ   # ≥ 1
    δ::Tδ   # > 0
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::SibuyaStoppedGamma)
    θ = float(D.θ); δ = float(D.δ)
    # T ~ Sibuya(1/θ) 
    t = rand(rng, Sibuya(1/θ))
    # M | T=t ~ Gamma(t/δ, 1)
    return rand(rng, Distributions.Gamma(t/δ, 1.0))
end