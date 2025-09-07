# --------- Frailty: positive-stable stopped gamma ----------
# T ~ Postable(α = 1/θ, scale = 1);   M | T=t  ~ Gamma(t/δ, 1)
struct PosStableStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ; δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::PosStableStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, PStable(inv(θ)))                # Postable(α=1/θ) con escala 1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))
end
