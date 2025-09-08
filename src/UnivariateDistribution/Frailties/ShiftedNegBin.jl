struct ShiftedNegBin{T} <: Distributions.DiscreteUnivariateDistribution
    r::T   # r = 1/θ
    p::T   # p = 1-π ∈ [0,1]
end
Distributions.rand(rng::Distributions.AbstractRNG, D::ShiftedNegBin) =
    D.r + rand(rng, Distributions.NegativeBinomial(D.r, D.p))
