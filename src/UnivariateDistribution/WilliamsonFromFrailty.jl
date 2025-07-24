struct WilliamsonFromFrailty{TF,d} <: Distributions.ContinuousUnivariateDistribution
    frailty_dist::TF
    function WilliamsonFromFrailty(frailty_dist,::Val{d}) where d
        return new{typeof(frailty_dist),d}(frailty_dist)
    end
    WilliamsonFromFrailty(frailty_dist,d) = WilliamsonFromFrailty(frailty_dist,Val(d))
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::WilliamsonFromFrailty{TF,d}) where {TF,d}
    f = rand(rng,D.frailty_dist)
    sy = rand(rng,Distributions.Erlang(d))
    return sy/f
end
function Distributions.cdf(D::WilliamsonFromFrailty{TF,d}, x::Real) where {TF,d}
    # how to compute this cdf ? 
    return 1 - Distributions.expectation(
        e -> Distributions.cdf(D.frailty_dist,e/x),
        Distributions.Erlang(d)
    )
end
function Distributions.pdf(D::WilliamsonFromFrailty{TF,d}, x::Real) where {TF,d}
    # how to compute this cdf ? 
    return 1/x^2 * Distributions.expectation(
        e -> Distributions.pdf(D.frailty_dist,e/x),
        Distributions.Erlang(d)
    )
end
Base.minimum(D::WilliamsonFromFrailty{TF,d}) where {TF,d} = 0
Base.maximum(D::WilliamsonFromFrailty{TF,d}) where {TF,d} = Inf