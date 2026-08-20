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
    isfinite(order) && order > 0 || throw(ArgumentError(
        "the Williamson order must be finite and positive",
    ))
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
    return Distributions.expectation(
        v -> Distributions.cdf(D.numerator, x * v), D.frailty_dist,
    )
end
function Distributions.pdf(D::WilliamsonFromFrailty, x::Real)
    x <= 0 && return zero(float(x))
    isinf(x) && return zero(float(x))
    return Distributions.expectation(
        v -> v * Distributions.pdf(D.numerator, x * v), D.frailty_dist,
    )
end
function Distributions.quantile(D::WilliamsonFromFrailty, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    lo = zero(float(p))
    hi = oftype(lo, Inf)
    iszero(p) && return lo
    isone(p) && return hi
    return Roots.find_zero(x -> Distributions.cdf(D, x) - p, (lo, hi))
end
Base.minimum(::WilliamsonFromFrailty) = 0
Base.maximum(::WilliamsonFromFrailty) = Inf

# Posterior frailty after observing Liouville coordinates whose Dirichlet
# parameters sum to `power` and whose radial coordinates sum to `shift`.
struct PowerTiltedFrailty{S<:Distributions.ValueSupport,TF,TP,TS,TN} <:
       Distributions.UnivariateDistribution{S}
    base::TF
    power::TP
    shift::TS
    normalizer::TN

    function PowerTiltedFrailty(base, power::Real, shift::Real)
        isfinite(power) && power >= 0 || throw(ArgumentError(
            "the frailty tilt power must be finite and non-negative",
        ))
        isfinite(shift) && shift >= 0 || throw(ArgumentError(
            "the frailty tilt shift must be finite and non-negative",
        ))
        normalizer = Distributions.expectation(
            v -> _power_tilt_weight(v, power, shift), base,
        )
        isfinite(normalizer) && normalizer > 0 || throw(ArgumentError(
            "the conditional frailty has zero or non-finite normalizing mass",
        ))
        support = Distributions.value_support(typeof(base))
        return new{support,typeof(base),typeof(power),typeof(shift),typeof(normalizer)}(
            base, power, shift, normalizer,
        )
    end
end


# Multiplying a Gamma(a, b) density by v^power * exp(-shift*v) gives another
# Gamma law, with shape a + power and rate inv(b) + shift. In particular this
# preserves Clayton's exact Beta-prime radial after Liouville conditioning.
function PowerTiltedFrailty(
    base::Distributions.Gamma, power::Real, shift::Real,
)
    isfinite(power) && power >= 0 || throw(ArgumentError(
        "the frailty tilt power must be finite and non-negative",
    ))
    isfinite(shift) && shift >= 0 || throw(ArgumentError(
        "the frailty tilt shift must be finite and non-negative",
    ))
    shape, scale = Distributions.params(base)
    return Distributions.Gamma(shape + power, inv(inv(scale) + shift))
end

function _power_tilt_weight(v, power, shift)
    v < 0 && return zero(float(v))
    iszero(v) && return iszero(power) ? one(float(v)) : zero(float(v))
    return exp(power * log(v) - shift * v)
end

Base.minimum(D::PowerTiltedFrailty) = Base.minimum(D.base)
Base.maximum(D::PowerTiltedFrailty) = Base.maximum(D.base)
function Distributions.expectation(f, D::PowerTiltedFrailty; kwargs...)
    return Distributions.expectation(D.base; kwargs...) do v
        f(v) * _power_tilt_weight(v, D.power, D.shift) / D.normalizer
    end
end
function Distributions.pdf(D::PowerTiltedFrailty, v::Real)
    return Distributions.pdf(D.base, v) *
           _power_tilt_weight(v, D.power, D.shift) / D.normalizer
end
function Distributions.cdf(D::PowerTiltedFrailty, x::Real)
    x < minimum(D) && return zero(float(x))
    x >= maximum(D) && return one(float(x))
    return Distributions.expectation(D.base) do v
        v <= x ? _power_tilt_weight(v, D.power, D.shift) / D.normalizer : zero(float(v))
    end
end
function Distributions.quantile(D::PowerTiltedFrailty, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    iszero(p) && return minimum(D)
    isone(p) && return maximum(D)
    lo = float(minimum(D))
    hi = float(maximum(D))
    if !isfinite(hi)
        hi = max(one(lo), lo + one(lo))
        while Distributions.cdf(D, hi) < p
            hi *= 2
        end
    end
    for _ in 1:64
        mid = (lo + hi) / 2
        Distributions.cdf(D, mid) < p ? (lo = mid) : (hi = mid)
    end
    return hi
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::PowerTiltedFrailty)
    if iszero(D.power)
        while true
            v = rand(rng, D.base)
            rand(rng) <= exp(-D.shift * v) && return v
        end
    elseif D.shift > 0
        logmax = D.power * (log(D.power / D.shift) - 1)
        while true
            v = rand(rng, D.base)
            logweight = iszero(v) ? -Inf : D.power * log(v) - D.shift * v
            log(rand(rng)) <= logweight - logmax && return v
        end
    end
    return Distributions.quantile(D, rand(rng))
end
