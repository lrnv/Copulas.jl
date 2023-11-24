# Corresponds to https://en.wikipedia.org/wiki/Logarithmic_distribution
struct Logarithmic{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    α::T # in (0,1), the weight of the logarithmic distribution, 1-p of the wikipedia parameter p. 
    h::T # = ln(1-α), so in (-Inf,0)
    function Logarithmic(h::T) where {T <: Real}
        Tf = promote_type(T,Float64)
        α = -expm1(h)
        return new{Tf}(Tf(α), Tf(h))
    end
end
Base.eltype(::Logarithmic{T}) where T = promote_type(T,Float64)
function Distributions.logpdf(d::Logarithmic{T}, x::Real) where T
    insupport(d, x) ? x*log1p(-d.α) - log(x) - log(-log(d.α)) : log(zero(T))
end
function _my_beta_inc(α,k)
    # this computes \int_{0}^(1-α) t^k/(1-t) dt, for α in [0,1] and k a positive integer. 
    r = zero(α)
    pwα = 1
    sgn = 1
    for l in 0:k
        r += sgn * binomial(k,l)  * (1 - pwα) / l
        sgn *= -1
        pwα *= α
    end
    return r
end
function Distributions.cdf(d::Logarithmic{T}, x::Real) where T
    # Needs cdf to be implemented here. 
    # quand even more quantile?
    # ee her e: https://en.wikipedia.org/wiki/Logarithmic_distribution 
    # return 1 + incbeta(p,k=1,0)/log1p(-p)
    k = Int(trunc(x))
    return 1 + _my_beta_inc(d.α,k)/log(d.α)
    # SpecialFunctions.beta_inc(k+1,0,p)/log1p(-p) *SpecialFunctions.beta(k+1,0)
end
function Distributions.quantile(d::Logarithmic{T}, p::Real) where T
    return Roots.find_zero(x -> Distributions.cdf(d,x) - p, (0, Inf))
end

function Distributions.rand(rng::Distributions.AbstractRNG, d::Logarithmic{T}) where T
    # Sample a Log(p) distribution with the algorithms "LK" and "LS" of Kemp (1981).
    if d.h > -3
        # Version "LS"
        t = -d.α / log1p(-d.α)
        u = rand(rng)
        p = t
        x = 1
        while u > p
            u = u - p
            x = x + 1
            p = p * d.α * (x-1) / x
        end
        return x
    else
        # Version "LK"
        u = rand(rng)
        if u > d.α
            return one(T)
        else
            v = rand(rng)
            h2 = d.h * v
            r = -expm1(h2)
            if u < r*r 
                lr = LogExpFunctions.log1mexp(h2)
                if iszero(lr)
                    return T(Inf)
                else
                    return floor(1 + log(u)/lr)
                end
            else
                if u > r 
                    return one(T)
                else
                    return T(2)
                end
            end
        end
    end
end