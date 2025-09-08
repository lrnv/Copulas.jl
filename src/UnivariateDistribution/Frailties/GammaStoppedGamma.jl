struct GammaStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ
    δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, Distributions.Gamma(1/θ, 1.0))        # shape=1/θ, scale=1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))     # shape=T/δ,  scale=1
end
