# See https://rdrr.io/rforge/copula/man/Sibuya.html

struct Sibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    p::T
    function Sibuya(p::T) where {T <: Real}
        @assert 0 < p ≤ 1
        new{T}(p)
    end
    Sibuya{T}(p) where T = Sibuya(T(p))
end
Base.minimum(::Sibuya) = 1
Base.maximum(::Sibuya) = Inf
function Distributions.rand(rng::Distributions.AbstractRNG, d::Sibuya{T}) where {T <: Real}
    u = rand(rng, T)
    if u <= d.p
        return T(1)
    end
    xMax = 1/eps(T)
    Ginv = ((1-u)*SpecialFunctions.gamma(1-d.p))^(-1/d.p)
    fGinv = floor(Ginv)
    if Ginv > xMax 
        return fGinv
    end
    if 1-u < 1/(fGinv*SpecialFunctions.beta(fGinv,1-d.p))
        return ceil(Ginv)
    end
    return fGinv
end
Distributions.mgf(D::Sibuya, t) = 1-(-expm1(t))^(D.p)
function Distributions.cdf(d::Sibuya, u::Real)
    u < 1 && return zero(float(u))
    isinf(u) && return one(float(u))
    d.p == 1 && return one(float(u))
    k = floor(Int, u)
    logtail = SpecialFunctions.loggamma(k + 1 - d.p) -
              SpecialFunctions.loggamma(1 - d.p) -
              SpecialFunctions.loggamma(k + 1)
    return -expm1(logtail)
end
function Distributions.logpdf(d::Sibuya, x::Real)
    Distributions.insupport(d, x) || return -Inf
    k = Int(x)
    d.p == 1 && return k == 1 ? zero(float(x)) : -Inf
    return log(d.p) + SpecialFunctions.loggamma(k - d.p) -
           SpecialFunctions.loggamma(1 - d.p) -
           SpecialFunctions.loggamma(k + 1)
end
