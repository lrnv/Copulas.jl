"""
        FlipDistortion(base)

Parameters
    * `base::Distortion` – underlying distortion

Survival / complement distortion: applies symmetry u ↦ 1-u to re-use existing
distortions when forming survival or flipped copulas.
"""
struct FlipDistortion{Disto} <: Distortion
    base::Disto
end
Distributions.cdf(D::FlipDistortion, u::Real) = 1.0 - Distributions.cdf(D.base, 1.0 - float(u))
Distributions.quantile(D::FlipDistortion, α::Real) = 1.0 - Distributions.quantile(D.base, 1.0 - float(α))

## Methods moved next to SurvivalCopula type
