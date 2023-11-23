"""
    LiouvilleCopula{d, T, TG}

Fields: 
* `α::NTuple{d,T}` : the weights for each dimension
* `G::TG` : the generator <: Generator. 

Constructor: 

    LiouvilleCopula(α::Vector{T},G::Generator)

The Liouville copula has a structure that resembles the [`ArchimedeanCopula`](@ref), when you look at it from it's radial-simplex decomposition. 

Recalling that, for C an archimedean copula with generator ``\\phi``, if ``\\mathbf U \\sim C``, then ``U \\equal R \\mathbf S`` for a random vector ``\\mathbf S \\sim`` `Dirichlet(ones(d))`, that is uniformity on the d-variate simplex, and a non-negative random variable ``R`` that is the Williamson d-transform of `\\phi`. 

The Liouville copula has exactly the same expression but using another Dirichlet distribution instead than uniformity over the simplex. 

References: 
* [mcneil2010](@cite) McNeil, A. J., & Nešlehová, J. (2010). From archimedean to liouville copulas. Journal of Multivariate Analysis, 101(8), 1772-1790.
"""
struct LiouvilleCopula{d,TG} <: Copula{d}
    α::NTuple{d,Int}
    G::TG
    function LiouvilleCopula(α::Vector{Int},G::Generator)
        d = length(α)
        @assert sum(α) <= max_monotony(G) "The generator you provided is not monotonous enough (the monotony index must be greater than sum(α), and thus this copula does not exists."
        return new{d, typeof(G)}(G)
    end
end
williamson_dist(C::LiouvilleCopula{d,TG}) where {d,TG} = williamson_dist(C.G,sum(C.α))

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T,CT<:LiouvilleCopula}
    # By default, we use the williamson sampling. 
    r = Distributions.rand(williamson_dist(C))
    for i in 1:length(C)
        x[i] = Distributions.rand(rng,Gamma(C.α[i],1))
        x[i] = x[i] * r
        x[i] = Distributions.cdf(williamson_dist(C.G,C.α[i]),x[i])
    end
    return x
end
function _cdf(C::CT,u) where {CT<:LiouvilleCopula}
    d = length(C)

    sx = sum(Distributions.quantile(williamson_dist(C.G,C.α[i]),u[i]) for i in 1:d)
    r = zero(eltype(u))
    for i in CartesianIndices(α)
        ii = (ij-1 for ij in Tuple(i))
        sii = sum(ii)
        fii = prod(factorial.(ii))
        # This version is really not efficient as derivatives of the generator ϕ⁽ᵏ⁾(C.G, sii, sx) could be pre-computed since many sii will be the same and sx does never change. 
        r += (-1)^sii / fii * ϕ⁽ᵏ⁾(C.G, sii, sx) * prod(x .^i)
    end
    return r
end
