# Type-preserving specialization for the scalar numerical inputs used by the
# generic Generator derivative machinery. TaylorSeries may omit trailing zero
# coefficients; padding them must preserve the coefficient representation.
function taylor(f::F, x₀::T, d::Int) where {F,T<:Real}
    # Seed the Taylor polynomial directly instead of forming `x₀ + Taylor1(...)`.
    # The latter can widen the series through generic scalar/series promotion
    # before `f` is evaluated. Direct coefficients preserve Float32, BigFloat,
    # and nested Real/AD representations supplied by the caller.
    seed = fill(zero(x₀), d + 1)
    seed[1] = x₀
    d > 0 && (seed[2] = one(x₀))
    rez = f(TaylorSeries.Taylor1(seed)).coeffs
    p = length(rez)
    p == d + 1 && return rez
    if p < d + 1
        v = fill(zero(eltype(rez)), d + 1)
        v[1:p] .= rez
        return v
    end
    return rez[1:(d + 1)]
end