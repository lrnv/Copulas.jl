"""
    ShiftedNegBin(r, p)

Parameters
    * `r > 0`
    * `p ∈ [0,1]`

Used as a discrete frailty; Laplace transform provides an Archimedean generator.

Shifted negative binomial: if `Y ∼ NegBin(r, p)` then `X = r + Y` has support {r, r+1, …}.
"""
struct ShiftedNegBin{T} <: Distributions.DiscreteUnivariateDistribution
    r::T   # r = 1/θ
    p::T   # p = 1-π ∈ [0,1]
end
Distributions.rand(rng::Distributions.AbstractRNG, D::ShiftedNegBin) =
    D.r + rand(rng, Distributions.NegativeBinomial(D.r, D.p))
