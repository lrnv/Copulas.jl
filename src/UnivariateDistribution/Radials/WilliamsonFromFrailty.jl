struct WilliamsonFromFrailty{TF,TN,TO} <: Distributions.ContinuousUnivariateDistribution
    frailty_dist::TF
    numerator::TN
    order::TO
    function WilliamsonFromFrailty(frailty_dist, order::Real)
        isfinite(order) && order > 0 || throw(ArgumentError(
            "the Williamson order must be finite and positive",
        ))
        numerator = Distributions.Gamma(order)
        return new{typeof(frailty_dist),typeof(numerator),typeof(order)}(
            frailty_dist, numerator, order,
        )
    end
end

# If V ~ Gamma(a, b), then Gamma(order, 1) / V is exactly
# BetaPrime(order, a) / b. Keep this ratio in closed form instead of evaluating
# expectations over the frailty for every cdf/pdf/quantile call.
function WilliamsonFromFrailty(frailty_dist::Distributions.Gamma, order::Real)
    isfinite(order) && order > 0 || throw(ArgumentError("the Williamson order must be finite and positive",))
    shape, scale = Distributions.params(frailty_dist)
    return inv(scale) * Distributions.BetaPrime(order, shape)
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::WilliamsonFromFrailty)
    f = rand(rng,D.frailty_dist)
    sy = rand(rng, D.numerator)
    return sy/f
end
function Distributions.cdf(D::WilliamsonFromFrailty, x::Real)
    x <= 0 && return zero(float(x))
    isinf(x) && return one(float(x))
    return Distributions.expectation(v -> Distributions.cdf(D.numerator, x * v), D.frailty_dist)
end
function Distributions.pdf(D::WilliamsonFromFrailty, x::Real)
    x <= 0 && return zero(float(x))
    isinf(x) && return zero(float(x))
    return Distributions.expectation(v -> v * Distributions.pdf(D.numerator, x * v), D.frailty_dist)
end
Distributions.logpdf(D::WilliamsonFromFrailty, x::Real) = log(Distributions.pdf(D, x))
function Distributions.quantile(D::WilliamsonFromFrailty, p::Real)
    return _quantile_from_cdf(D, p)
end
Base.minimum(::WilliamsonFromFrailty) = 0
Base.maximum(::WilliamsonFromFrailty) = Inf
