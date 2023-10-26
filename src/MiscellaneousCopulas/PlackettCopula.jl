#= Details about Plackett copulation are found in Joe, H. (2014). 
   Dependence modeling with copulas. CRC press, Page.164
==#

#Create an instance of the Plackett copula
struct PlackettCopula{P} <: Copula{2} # since it is only bivariate.
    θ::P  # Copula parameter

    function PlackettCopula(θ) where {P}
        if θ == 1
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

# Base.length(S::PlackettCopula{P}) where {P} = 2 # length should return the dimension of the copula, bnut i think it is already working without this definition.
Base.eltype(S::PlackettCopula{P}) where {P} = P # this shuold be P. 

# CDF calculation for bivariate Plackett Copula
function Distributions.cdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = 1 + η * (u + v)
    term2 = sqrt(term1^2 - 4 * S.θ * η * u * v)
    return 0.5 * η^(-1) * (term1 - term2)
end

# PDF calculation for bivariate Plackett Copula
function Distributions._logpdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = S.θ * (1 + η * (u + v - 2 * u * v))
    term2 = (1+η*(u+v))^2-4*(S.θ)*η*u*v
    return log(term1) - 3 * log(term2)/2 # since we are supposed to return the logpdf. 
end
import Random

#=  Details about the algorithm to generate copula samples 
    can be seen in the following references
    Johnson, Mark E. Multivariate statistical simulation:
    A guide to selecting and generating continuous multivariate distributions.
    Vol. 192. John Wiley & Sons, 1987. Page 193.
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.38.
==#

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:PlackettCopula}
    u = rand(rng)
    t = rand(rng)
    a = t * (1 - t)
    b = c.θ + a * (c.θ - 1)^2
    cc = 2a * (u * c.θ^2 + 1 - u) + c.θ * (1 - 2a)
    d = sqrt(c.θ) * sqrt(c.θ + 4a * u * (1 - u) * (1 - c.θ)^2)
    v = (cc - (1 - 2t) * d) / (2b)
    x[1] = u
    x[2] = v
    return x
end
function Base.rand(rng::Distributions.AbstractRNG,C::CT) where CT<: PlackettCopula
    x = rand(rng,length(C))
    u = rand(rng)
    t = rand(rng)
    a = t * (1 - t)
    b = c.θ + a * (c.θ - 1)^2
    cc = 2a * (u * c.θ^2 + 1 - u) + c.θ * (1 - 2a)
    d = sqrt(c.θ) * sqrt(c.θ + 4a * u * (1 - u) * (1 - c.θ)^2)
    v = (cc - (1 - 2t) * d) / (2b)
    x[1] = u
    x[2] = v
    return x
end

# Calculate Spearman's rho based on the PlackettCopula parameters
function ρ(c::PlackettCopula{P}) where P
    return (c.θ+1)/(c.θ-1)-(2*c.θ*log(c.θ)/(c.θ-1)^2)
end