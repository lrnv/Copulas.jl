struct WCopula{d} <: Copula{d} end
WCopula(d) = WCopula{d}()
Distributions.cdf(::WCopula{d},u) where {d} = max(1 + sum(u)-d,0)
function Distributions._rand!(rng::Distributions.AbstractRNG, ::WCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    @assert d==2
    x[1] = rand(rng)
    x[2] = 1-x[1] 
end
τ(::WCopula{d}) where d = -1/(d-1)