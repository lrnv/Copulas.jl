using Distributions, Copulas
#= Details about Plackett copulation are found in Joe, H. (2014). 
   Dependence modeling with copulas. CRC press, Page.164
==#

#Create an instance of the Plackett copula
struct PlackettCopula{P} <: ContinuousMultivariateDistribution
    θ::P  # Copula parameter

    function PlackettCopula(θ) where {P}
        if θ == 1.0
            return IndependentCopula(2)
        elseif θ == 0
            return MCopula(2)
        elseif θ == Inf
            return WCopula(2)
        else
            θ >= 0 || throw(ArgumentError("Theta must be non-negative"))
            return new{typeof(θ)}(θ)
        end
    end
end

Base.length(S::PlackettCopula{P}) where {P} = 1
Base.eltype(S::PlackettCopula{P}) where {P} = Float64

import Distributions: cdf, pdf

# CDF calculation for bivariate Plackett Copula
function cdf(S::PlackettCopula{P}, uv::Tuple{Float64, Float64}) where {P}
    u, v = uv
    η = S.θ - 1

    if S.θ == 1.0
        return cdf(IndependentCopula(2), uv)
    elseif S.θ == Inf
        return cdf(WCopula(2), uv)
    elseif S.θ == 0
        return cdf(MCopula(2), uv)
    else
        term1 = 1 + η * (u + v)
        term2 = sqrt(term1^2 - 4 * S.θ * η * u * v)
        return 0.5 * η^(-1) * (term1 - term2)
    end
end

# PDF calculation for bivariate Plackett Copula
function pdf(S::PlackettCopula{P}, uv::Tuple{Float64, Float64}) where {P}
    u, v = uv
    η = S.θ - 1

    if S.θ == 1.0
        return pdf(IndependentCopula(2), uv)
    elseif S.θ == Inf
        throw(ArgumentError("Indefinite density"))
    elseif S.θ == 0
        throw(ArgumentError("Indefinite density"))
    else
        term1 = S.θ * (1 + η * (u + v - 2 * u * v))
        term2 = (1+η*(u+v))^2-4*(S.θ)*η*u*v
        return term1/term2^(3/2)
    end
end
import Random

#=  Details about the algorithm to generate copula samples 
    can be seen in the following references
    Johnson, Mark E. Multivariate statistical simulation:
    A guide to selecting and generating continuous multivariate distributions.
    Vol. 192. John Wiley & Sons, 1987. Page 193.
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.38.
==#

# Copula random sample simulator
function rand(rng::Random.AbstractRNG, c::PlackettCopula{P}, n::Int) where P
    samples = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        u = rand(rng)
        t = rand(rng)
        a = t * (1 - t)
        b = c.θ + a * (c.θ - 1)^2
        cc = 2a * (u * c.θ^2 + 1 - u) + c.θ * (1 - 2a)
        d = sqrt(c.θ) * sqrt(c.θ + 4a * u * (1 - u) * (1 - c.θ)^2)
        v = (cc - (1 - 2t) * d) / (2b)
        samples[1, i] = u
        samples[2, i] = v
    end
    return samples
end

# Calculate Spearman's rho based on the PlackettCopula parameters
function spearman_rho(c::PlackettCopula{P}) where P
     rho = (c.θ+1)/(c.θ-1)-(2*c.θ*log(c.θ)/(c.θ-1)^2)
     return rho
end