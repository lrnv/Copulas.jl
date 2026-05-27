# =============================================================================
# Taylor-series-compatible ϕ / ϕ⁻¹ methods for the Frank generator.
#
# The nested-Archimedean density recursion evaluates the generator on a
# `TaylorSeries.Taylor1` argument (to obtain ϕ⁽ᵏ⁾ and the composition Taylor
# coefficients). For θ > 0, Frank's default ϕ / ϕ⁻¹ are written with
# `LogExpFunctions.log1mexp`, which has no `Taylor1` method, so the nested path
# would otherwise fail for Frank.
#
# These methods add the *equivalent* closed forms built only from
# `exp` / `expm1` / `log` / `log1p` (all Taylor1-compatible). They are exactly
# the forms Copulas already uses for the θ < 0 branch, extended here to every
# θ ≠ 0 and specialised on a `Taylor1` argument:
#
#     ϕ(t)    = -log1p( e^{-t} · expm1(-θ) ) / θ
#     ϕ⁻¹(t)  = -log( expm1(-θ t) / expm1(-θ) )
#
# They only ADD methods on a `Taylor1` argument; the scalar Float64/BigFloat ϕ
# path is untouched, so Frank's ordinary behaviour is unchanged.
# =============================================================================

function ϕ(G::FrankGenerator, t::TaylorSeries.Taylor1{T}) where {T}
    θ = T(G.θ)
    return -log1p(exp(-t) * expm1(-θ)) / θ
end

function ϕ⁻¹(G::FrankGenerator, t::TaylorSeries.Taylor1{T}) where {T}
    θ = T(G.θ)
    return -log(expm1(-θ * t) / expm1(-θ))
end
