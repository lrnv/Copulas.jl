abstract type EllipticalCopula{d,MT} <: Copula{d} end
Base.eltype(C::CT) where CT<:EllipticalCopula = Base.eltype(N(CT)(C.Σ))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT <: EllipticalCopula}
    Random.rand!(rng,N(CT)(C.Σ),x)
    x .= cdf.(U(CT),x)
end
function Distributions._logpdf(C::CT, u) where {CT <: EllipticalCopula}
    x = quantile.(U(CT),u)
    return Distributions.logpdf(N(CT)(C.Σ),x) - sum(Distributions.logpdf.(U(CT),x))
end
function make_cor!(Σ)
    # Verify that Σ is a correlation matrix, otherwise make it so : 
    d = size(Σ,1)
    σ = [1/sqrt(Σ[i,i]) for i in 1:d]
    for i in 1:d
        for j in 1:d
            Σ[i,j] *= σ[i] .* σ[j]
        end
    end
end

# Deprecated in favor of the Hcubature version that is more generic
# function Distributions.cdf(C::CT,u) where {CT<:EllipticalCopula} 
#     @assert length(C) == length(u) 
#     x = quantile.(U(CT),u)
#     return Distributions.cdf(N(CT)(C.Σ),x)
# end