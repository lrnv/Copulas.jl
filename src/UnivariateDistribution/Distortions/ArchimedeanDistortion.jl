###########################################################################
#####  ArchimedeanCopula fast-paths
###########################################################################
struct ArchimedeanDistortion{TG, T} <: Distortion
    G::TG
    p::Int
    sJ::T
    den::T
    function ArchimedeanDistortion(G::TG, p::Int, sJ::S, den::T) where {S<:Real,T<:Real,TG}
        sJ, den = promote(sJ, den)
        return new{TG,typeof(sJ)}(G, p, sJ, den)
    end
end
function Distributions.cdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    R = float(promote_type(typeof(u), T))
    u <= 0 && return zero(R)
    u >= 1 && return one(R)
    return ϕ⁽ᵏ⁾(D.G, D.p, D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
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
## `conditional_copula` specialization is defined next to ArchimedeanCopula.
function Distributions.logpdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    0 <= u <= 1 || return float(promote_type(typeof(u), T))(-Inf)
    ξ = ϕ⁻¹(D.G, float(u))
    num = ϕ⁽ᵏ⁾(D.G, D.p + 1, D.sJ + ξ)
    return log(abs(num)) - log(abs(D.den)) - log(abs(ϕ⁽ᵏ⁾(D.G, 1, ξ)))
end
