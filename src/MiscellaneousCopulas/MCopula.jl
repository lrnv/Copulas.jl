struct MCopula{d} <: Copula{d} end
MCopula(d) = MCopula{d}()
Distributions.cdf(::MCopula{d},u) where {d} = minimum(u)
function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
τ(::MCopula) = 1