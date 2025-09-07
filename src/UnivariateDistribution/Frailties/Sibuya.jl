# See https://rdrr.io/rforge/copula/man/Sibuya.html

struct Sibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    p::T
    function Sibuya(p::T) where {T <: Real}
        @assert 0 < p ≤ 1
        new{T}(p)
    end
    Sibuya{T}(p) where T = Sibuya(T(p))
end
Base.minimum(::Sibuya) = 0
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
    k = trunc(u)
    return 1 - abs(binom(d.p-1, k))
end
function Distributions.logpdf(d::Sibuya, x::Real)
    insupport(d, x) ? log(abs(binom(d.p, k))) : -Inf
end