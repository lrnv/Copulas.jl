struct MixtureCopula{d,CT1,CT2,T}<:Copula{d} where {CT1 <: Copula{d}, CT2 <: Copula{d}, T}
    C1::CT1
    C2::CT2
    p::T
    function MixtureCopula(C₁, C₂, w₁, w₂)
        @assert w₁ >= 0 && w₂ >= 0
        p = w₁ / (w₁ + w₂)
        if p >= 0.5
            new{eltype(C₁),eltype(C₂), eltype(p)}(C₁,C₂,p)
        else
            new{eltype(C₂),eltype(C₁), eltype(p)}(C₂,C₁,1-p)
        end
    end
end

function MixtureCopula(CopList,weights)
    n = length(CopList)
    if n ==1
        return CopList[1]
    elseif n == 2
        return MixtureCopula(CopList[1],CopList[2], weights[1],weights[2])
    else
        return MixtureCopula(
            CopList[1],
            MixtureCopula(CopList[2:end],weights[2:end]),
            weights[1],
            sum(weights[2:end])
        )
    end
end


function Distributions.cdf(C::MixtureCopula,u)
    return p * Distributions.cdf(C.C1,u) + (1-p) * Distributions.cdf(C.C2,u)
end



function Distributions._rand!(rng::Distributions.AbstractRNG, C::MixtureCopula, x::AbstractVector{T}) where T<:Real
    if rand() < p
        Distributions._rand!(rng,C.C1,x)
    else
        Distribution._rand!(rng,C.C2,x)
    end
end
function Base.rand(rng::Distributions.AbstractRNG,C::MixtureCopula)
    x = zeros(d)
    Distributions._rand!(C,x)
end