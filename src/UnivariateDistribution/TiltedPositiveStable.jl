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