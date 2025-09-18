"""
    PosStableStoppedGamma(θ, δ)

Parameters
    * `θ > 0`
    * `δ > 0`

Used as frailty for Archimedean copulas (positive stable stopped by gamma).

Hierarchical frailty:
```math
T \\sim \\mathrm{PositiveStable}(1/θ, 1), \\qquad X \\mid T=t \\sim \\mathrm{Gamma}(t/δ, 1).
```
θ, δ > 0. Combines heavy-tailed positive stable mixing with a gamma conditional law.
"""
struct PosStableStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ; δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::PosStableStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, PStable(inv(θ)))                # Postable(α=1/θ) con escala 1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))
end
