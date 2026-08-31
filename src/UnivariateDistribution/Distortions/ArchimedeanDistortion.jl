###########################################################################
#####  ArchimedeanCopula fast-paths
###########################################################################
struct ArchimedeanDistortion{TG, T} <: Distortion
    G::TG
    p::Int
    sJ::T
    den::T
    ArchimedeanDistortion(G::TG, p::Int, sJ::T, den::T) where {T<:Real, TG} = new{TG, T}(G, p, sJ, den)
end
function Distributions.cdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    R = float(promote_type(typeof(u), T))
    u <= 0 && return zero(R)
    u >= 1 && return one(R)
    return ϕ⁽ᵏ⁾(D.G, D.p, D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
distortion_measure_style(D::ArchimedeanDistortion{<:WilliamsonGenerator}) =
    archimedean_measure_style(D.G, Val(D.p + 1))
function Distributions.logcdf(D::ArchimedeanDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(D.sJ), typeof(D.den)))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    ξ = ϕ⁻¹(D.G, T(u))
    num = ϕ⁽ᵏ⁾(D.G, D.p, T(D.sJ) + ξ)
    return log(abs(num)) - log(abs(T(D.den)))
end
function Distributions.quantile(D::ArchimedeanDistortion{TG, T}, α::Real) where {TG, T}
    y = ϕ⁽ᵏ⁾⁻¹(D.G, D.p, α * D.den; start_at = D.sJ)
    return ϕ(D.G, y - D.sJ)
end
function Distributions.quantile(
    D::ArchimedeanDistortion{<:WilliamsonGenerator},
    α::Real,
)
    distortion_measure_style(D) isa NonAbsolutelyContinuousMeasure &&
        return _unit_quantile(D, α)
    return invoke(
        Distributions.quantile,
        Tuple{ArchimedeanDistortion,Real},
        D,
        α,
    )
end
## ConditionalCopula moved next to ArchimedeanCopula definition
function Distributions.logpdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    0 <= u <= 1 || return float(promote_type(typeof(u), T))(-Inf)
    ξ = ϕ⁻¹(D.G, float(u))
    num = ϕ⁽ᵏ⁾(D.G, D.p + 1, D.sJ + ξ)
    return log(abs(num)) - log(abs(D.den)) - log(abs(ϕ⁽ᵏ⁾(D.G, 1, ξ)))
end
