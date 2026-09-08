abstract type QuantileStrategy end
struct CDFQuantile <: QuantileStrategy end
struct LogCDFQuantile <: QuantileStrategy end

quantile_strategy(::Type) = CDFQuantile()

@inline _quantile_objective(::CDFQuantile, d, x) = Distributions.cdf(d, x)
@inline _quantile_objective(::LogCDFQuantile, d, x) = Distributions.logcdf(d, x)
@inline _quantile_target(::CDFQuantile, p) = p
@inline _quantile_target(::LogCDFQuantile, p) = log(p)

function _quantile_at_least(strategy, d, x, target)
    value = _quantile_objective(strategy, d, x)
    isnan(value) && throw(ArgumentError("the CDF returned NaN at x = $x"))
    return value >= target
end

"""
    _quantile_from_cdf([strategy], d, p)

Compute the generalized quantile `inf(x: cdf(d, x) >= p)`. Finite support
bounds are taken from `minimum(d)` and `maximum(d)`; infinite bounds are
bracketed geometrically. `LogCDFQuantile()` selects `logcdf` as the monotone
objective when ordinary CDF values may underflow.
"""
_quantile_from_cdf(d, p::Real) =
    _quantile_from_cdf(quantile_strategy(typeof(d)), d, p)

function _quantile_from_cdf(strategy::QuantileStrategy, d, p::Real)
    T = typeof(float(p))
    zero(T) <= p <= one(T) || throw(ArgumentError("p must be between 0 and 1"))

    lower = T(minimum(d))
    upper = T(maximum(d))
    iszero(p) && return lower
    isone(p) && return upper
    target = _quantile_target(strategy, T(p))

    if isfinite(lower)
        _quantile_at_least(strategy, d, lower, target) && return lower
    else
        lower = -one(T)
        while _quantile_at_least(strategy, d, lower, target)
            next = lower * T(2)
            isfinite(next) || throw(ArgumentError("could not establish a finite lower quantile bracket"))
            lower = next
        end
    end

    if isfinite(upper)
        _quantile_at_least(strategy, d, upper, target) ||
            throw(ArgumentError("the CDF does not reach p = $p on its support"))
    else
        upper = max(one(T), lower + one(T))
        while !_quantile_at_least(strategy, d, upper, target)
            next = upper > zero(T) ? upper * T(2) : one(T)
            isfinite(next) || throw(ArgumentError("could not establish a finite upper quantile bracket"))
            upper = next
        end
    end

    while true
        middle = lower / T(2) + upper / T(2)
        if middle == lower || middle == upper
            predecessor = prevfloat(upper)
            if predecessor > lower &&
                    _quantile_at_least(strategy, d, predecessor, target)
                upper = predecessor
                continue
            end
            return upper
        end
        if _quantile_at_least(strategy, d, middle, target)
            upper = middle
        else
            lower = middle
        end
    end
end
