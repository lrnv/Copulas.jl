struct SibuyaStoppedPosStable{T} <: Distributions.ContinuousUnivariateDistribution
    θ::T   # ≥ 1
    δ::T   # ≥ 1
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::SibuyaStoppedPosStable)
    θ = float(D.θ); δ = float(D.δ)
    t = rand(rng, Sibuya(1/θ))                 # T ~ Sibuya(1/θ)
    return rand(rng, PStable(1/δ; scale = t^δ))# M | T=t ~ Postable(α=1/δ, scale=t^δ)
end