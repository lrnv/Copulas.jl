"""
    TiltedPositiveStable(α, λ)

Parameters
    * `α ∈ (0,1]`
    * `λ ≥ 0`

Used as a tilted radial law for Archimedean copulas (producing alternative
Williamson transforms when λ>0).

Exponential tilt of a positive α‑stable: if `X ~ PStable(α)` then the tilted law
has density proportional to `e^{-λ x} f_{X}(x)`.
"""
struct TiltedPositiveStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T   # 0<α<=1
    λ::T   # λ>=0
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::TiltedPositiveStable)
    α, λ = float(D.α), float(D.λ)
    while true
        x = rand(rng, PStable(α))                  # X ~ Postable(α)
        if rand(rng) ≤ exp(-λ*x)                   # acepta con prob e^{-λ X}
            return x                               # ← M
        end
    end
end
