# Type-preserving specialization for the scalar numerical inputs used by the
# generic Generator derivative machinery. TaylorSeries may omit trailing zero
# coefficients; padding them must preserve the coefficient representation.
function taylor(f::F, x₀::Real, d::Int) where {F}
    rez = f(x₀ + TaylorSeries.Taylor1(typeof(x₀), d)).coeffs
    p = length(rez)
    p == d + 1 && return rez
    if p < d + 1
        v = fill(zero(eltype(rez)), d + 1)
        v[1:p] .= rez
        return v
    end
    return rez[1:(d + 1)]
end
