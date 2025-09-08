struct GammaStoppedPositiveStable{Tα,Tβ} <: Distributions.ContinuousUnivariateDistribution
    α::Tα   # = 1/δ
    β::Tβ   # = 1/θ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedPositiveStable)
    α, β = float(D.α), float(D.β)
    T = rand(rng, Distributions.Gamma(β, 1.0))        # Gamma(shape=β, scale=1)
    return rand(rng, PStable(α; scale = T^(1/α)))     # Z(T)
end
