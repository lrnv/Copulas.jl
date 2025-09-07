# Corresponds to https://en.wikipedia.org/wiki/Logarithmic_distribution
struct Logarithmic{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    α::T # in (0,1), the weight of the logarithmic distribution.
    h::T # = ln(1-α), so in (-Inf,0)
    function Logarithmic(h::T) where {T <: Real}
        Tf = promote_type(T,Float64)
        α = -expm1(h)
        return new{Tf}(Tf(α), Tf(h))
    end
    Logarithmic{T}(h) where T = Logarithmic(T(h))
end
Base.eltype(::Logarithmic{T}) where T = promote_type(T,Float64)
function Distributions.logpdf(d::Logarithmic{T}, x::Real) where T
    insupport(d, x) ? x*log1p(-d.α) - log(x) - log(-log(d.α)) : log(zero(T))
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