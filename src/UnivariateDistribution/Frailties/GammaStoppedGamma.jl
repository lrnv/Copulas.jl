"""
    GammaStoppedGamma(θ, δ)

Parameters
    * `θ > 0` – outer inverse‑shape parameter (heavier tail when larger)
    * `δ > 0` – inner scaling parameter

Used as frailty for Archimedean generators through `ϕ(t)=E[e^{-tX}]`.

Positive frailty distribution defined hierarchically:
```math
T \\sim \\text{Gamma}(1/θ, 1) \\quad (\\text{shape}=1/θ,\\ \\text{scale}=1),\\\\
X \\mid T \\sim \\text{Gamma}(T/δ, 1),
```
so the observed variable is `X`. Both θ, δ > 0. The Laplace / mgf does not have a
simple closed form, but simulation is straightforward via two gamma draws.


See also: [`FrailtyGenerator`](@ref).
"""
struct GammaStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ
    δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, Distributions.Gamma(1/θ, 1.0))        # shape=1/θ, scale=1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))     # shape=T/δ,  scale=1
end
