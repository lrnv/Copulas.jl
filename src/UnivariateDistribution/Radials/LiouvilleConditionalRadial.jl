# Radial representation of a fractional tilt. For X = R * D_α and fixed
# X_J = x_J, the sum S of the remaining coordinates has density proportional
# to f_R(S+s_J) * (S+s_J)^(1-α₀) * S^(α_I-1).
struct LiouvilleConditionalRadial{TR,TS,TA0,TAI,TN} <: Distributions.ContinuousUnivariateDistribution
    radial::TR
    shift::TS
    source_order::TA0
    target_order::TAI
    normalizer::TN
    integration_knots::Vector{TN}
    cumulative_masses::Vector{TN}

    function LiouvilleConditionalRadial(radial::TR, shift::TS, source_order::TA0, target_order::TAI) where {TR,TS,TA0,TAI}
        upper = Base.maximum(radial) - shift
        upper > 0 || throw(ArgumentError("the conditioning point is outside the radial support"))
        T = typeof(float(shift + source_order + target_order))
        transformed_kernel(t) = _liouville_conditional_transformed_kernel(radial, shift, source_order, target_order, upper, t)
        normalizer, _, segments = QuadGK.quadgk_segbuf(transformed_kernel, zero(T), one(T); rtol=sqrt(eps(T)))
        isfinite(normalizer) && normalizer > 0 || throw(ArgumentError("the conditioning event has zero or non-finite density"))
        sort!(segments; by=segment -> segment.a)
        integration_knots = [first(segments).a; map(segment -> segment.b, segments)]
        cumulative_masses = [zero(normalizer); cumsum(map(segment -> segment.I, segments))]
        cumulative_masses[end] = normalizer
        return new{TR,TS,TA0,TAI,typeof(normalizer)}(
            radial, shift, source_order, target_order, normalizer,
            integration_knots, cumulative_masses,
        )
    end
end

function _liouville_conditional_transformed_kernel(
    radial, shift, source_order, target_order, upper, t,
)
    if isfinite(upper)
        s = upper * t
        jacobian = upper
    else
        denominator = 1 - t
        s = t / denominator
        jacobian = inv(denominator^2)
    end
    return _liouville_conditional_kernel(
        radial, shift, source_order, target_order, s,
    ) * jacobian
end

function _liouville_conditional_unit_coordinate(D::LiouvilleConditionalRadial, s)
    upper = maximum(D)
    return isfinite(upper) ? s / upper : s / (1 + s)
end

function _liouville_conditional_radial_coordinate(D::LiouvilleConditionalRadial, t)
    upper = maximum(D)
    return isfinite(upper) ? upper * t : t / (1 - t)
end

function _liouville_conditional_cached_integral(D::LiouvilleConditionalRadial, t)
    knots = D.integration_knots
    masses = D.cumulative_masses
    segment = min(searchsortedlast(knots, t), length(knots) - 1)
    base = masses[segment]
    t == knots[segment] && return base
    upper = maximum(D)
    partial = QuadGK.quadgk(
        x -> _liouville_conditional_transformed_kernel(
            D.radial, D.shift, D.source_order, D.target_order, upper, x,
        ),
        knots[segment], t;
        rtol=sqrt(eps(typeof(float(t)))),
    )[1]
    return base + partial
end

function _liouville_conditional_kernel(radial, shift, source_order, target_order, s)
    s < 0 && return zero(float(s))
    r = s + shift
    return Distributions.pdf(radial, r) * r^(1 - source_order) * s^(target_order - 1)
end

Base.minimum(D::LiouvilleConditionalRadial) = zero(float(D.shift))
Base.maximum(D::LiouvilleConditionalRadial) = Base.maximum(D.radial) - D.shift
function Distributions.pdf(D::LiouvilleConditionalRadial, s::Real)
    minimum(D) <= s <= maximum(D) || return zero(float(s))
    return _liouville_conditional_kernel(
        D.radial, D.shift, D.source_order, D.target_order, s,
    ) / D.normalizer
end
Distributions.logpdf(D::LiouvilleConditionalRadial, s::Real) = log(Distributions.pdf(D, s))
function Distributions.cdf(D::LiouvilleConditionalRadial, s::Real)
    s <= minimum(D) && return zero(float(s))
    s >= maximum(D) && return one(float(s))
    t = _liouville_conditional_unit_coordinate(D, s)
    value = _liouville_conditional_cached_integral(D, t) / D.normalizer
    return clamp(value, zero(value), one(value))
end
function Distributions.quantile(D::LiouvilleConditionalRadial, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    iszero(p) && return minimum(D)
    isone(p) && return maximum(D)
    target = p * D.normalizer
    segment = min(
        searchsortedlast(D.cumulative_masses, target),
        length(D.integration_knots) - 1,
    )
    a, b = D.integration_knots[segment], D.integration_knots[segment + 1]
    base, total = D.cumulative_masses[segment], D.cumulative_masses[segment + 1]
    objective(t) = t == a ? base - target :
                   t == b ? total - target :
                   _liouville_conditional_cached_integral(D, t) - target
    t = Roots.find_zero(objective, (a, b), Roots.Brent())
    return _liouville_conditional_radial_coordinate(D, t)
end
Distributions.rand(rng::Distributions.AbstractRNG, D::LiouvilleConditionalRadial) =
    Distributions.quantile(D, rand(rng))
