"""
    NestedDistortion{TC,T} <: Distortion

Closed-form conditional marginal `U_i | U_js = u_js` of a
[`NestedArchimedeanCopula`](@ref). Its `cdf(D, u_i)` is the mixed partial of the
nested CDF over the conditioned set `js` (with `i` and every other coordinate
entering only as CDF arguments), divided by the observed-marginal density
`c_O = pdf(subsetdims(C, js), u_js)`. This routes the numerator through our
O(d²) Faà di Bruno tree walk rather than ForwardDiff. Handles general `p`, so it
is reused for each per-coordinate distortion the generic `ConditionalCopula`
constructor builds in the multi-unobserved case.
"""
struct NestedDistortion{TC,T} <: Distortion
    C::TC
    i::Int
    js::Vector{Int}
    ujs::Vector{T}
    logden::T
    utemplate::Vector{T}
    cdfcensored::Vector{Bool}
    pdfcensored::Vector{Bool}
    function NestedDistortion(C, i, js, ujs, logden, utemplate,
                              cdfcensored, pdfcensored)
        T = promote_type(eltype(ujs), typeof(logden), eltype(utemplate))
        return new{typeof(C),T}(
            C, i, collect(Int, js), T.(ujs), T(logden), T.(utemplate),
            collect(Bool, cdfcensored), collect(Bool, pdfcensored),
        )
    end
end

function Distributions.logcdf(D::NestedDistortion, ui::Real)
    # Boundary guards keep the generic Distortion.quantile bisection well-posed
    # and logcdf monotone: P(U_i ≤ 0 | ·) = 0 ⇒ logcdf = -Inf; P(U_i ≤ 1 | ·) = 1
    # ⇒ logcdf = 0.
    ui <= 0 && return -Inf
    ui >= 1 && return 0.0
    T = float(promote_type(typeof(ui), Float64))
    u = [k == D.i ? T(ui) : T(D.utemplate[k]) for k in eachindex(D.utemplate)]
    # Observed/differentiated = js only; dim i AND every other unobserved coord
    # are censored (enter the argument-sum only, no differentiation).
    return _censored_copula_logpdf(D.C, u, D.cdfcensored, T) - D.logden
end

Distributions.cdf(D::NestedDistortion, ui::Real) = exp(Distributions.logcdf(D, ui))

function Distributions.logpdf(D::NestedDistortion, ui::Real)
    T = float(promote_type(typeof(ui), Float64))
    zero(T) < ui < one(T) || return T(-Inf)

    u = [k == D.i ? T(ui) : T(D.utemplate[k]) for k in eachindex(D.utemplate)]

    # Differentiate the nested CDF with respect to both the conditioning
    # coordinates and the free coordinate. All other coordinates stay
    # marginalised at one.
    return _censored_copula_logpdf(D.C, u, D.pdfcensored, T) - T(D.logden)
end
