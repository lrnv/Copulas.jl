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