struct WilliamsonFromFrailty{TF,d} <: Distributions.ContinuousUnivariateDistribution
    frailty_dist::TF
    function WilliamsonFromFrailty(frailty_dist,d)
        return new{typeof(frailty_dist),d}(frailty_dist)
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::WilliamsonFromFrailty{TF,d}) where {TF,d}
    f = rand(rng,D.frailty_dist)
    sy = rand(rng,Distributions.Erlang(d))
    return sy/f
end
function _p(D::WilliamsonFromFrailty{TF,d}, x) where {TF,d}

    # Maybe we need to use a taylor serie approximation near x == 0
    # since this is not working for x too small. 
    # otherwise, we could simply truncate this: 
    if x < sqrt(eps(eltype(x)))
        return one(x)
    end

    return Distributions.expectation(
        e -> Distributions.cdf(D.frailty_dist,e/x),
        Distributions.Erlang(d)
    )
end
function Distributions.cdf(D::WilliamsonFromFrailty{TF,d}, x::Real) where {TF,d}
    # how to compute this cdf ? 
    return 1 - _p(D,x)
end
function Distributions.quantile(D::WilliamsonFromFrailty{TF,d},u::Real) where {TF,d}
    return Roots.find_zero(x -> _p(D, x) - (1-u), (0, Inf))
end
function Distributions.pdf(D::WilliamsonFromFrailty{TF,d}, x::Real) where {TF,d}
    # how to compute this cdf ? 
    return 1/x^2 * Distributions.expectation(
        e -> Distributions.pdf(D.frailty_dist,e/x),
        Distributions.Erlang(d)
    )
end