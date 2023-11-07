struct WilliamsonFromFrailty{TF,d} <: Distributions.ContinuousUnivariateDistribution
    frailty_dist::TF
    function WilliamsonFromFrailty(frailty_dist,d)
        return new{typeof(frailty_dist),d}(frailty_dist)
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::WilliamsonFromFrailty{TF,d}) where {TF,d}
    f = rand(rng,D.frailty_dist)
    sy = sum(.-log.(rand(rng,d)))
    return sy/f
end