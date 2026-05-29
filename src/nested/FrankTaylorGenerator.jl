# =============================================================================
# Taylor-series-compatible ϕ / ϕ⁻¹ methods for the Frank generator.
#
# `_composition_taylor` in `NestedArchimedeanDensity.jl` evaluates the inner-to-
# outer change of variables ϕ⁻¹_outer ∘ ϕ_inner on a `TaylorSeries.Taylor1`
# argument via Copulas' generic `taylor(f, x₀, d)` primitive. For θ > 0, Frank's
# default ϕ / ϕ⁻¹ are written with `LogExpFunctions.log1mexp`, which has no
# `Taylor1` method, so the nested path would otherwise fail when Frank appears
# anywhere in the tree. (The k-th derivative ϕ⁽ᵏ⁾ of Frank itself is upstream's
# closed-form PolyLog implementation in `src/Generator/FrankGenerator.jl` and
# does not touch a Taylor expansion.)
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
