struct WilliamsonCopula{d,Tϕ,TX} <: ArchimedeanCopula{d}
    ϕ::Tϕ
    X::TX
end
function WilliamsonCopula(X::Distributions.UnivariateDistribution, d)
    ϕ = WilliamsonTransforms.𝒲(X,d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
function WilliamsonCopula(ϕ::Function, d)
    X = WilliamsonTransforms.𝒲₋₁(ϕ,d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
function WilliamsonCopula(ϕ::Function, X::Distributions.UnivariateDistribution, d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:WilliamsonCopula}
    r = rand(rng,C.X)
    Random.rand!(rng,x)
    for i in 1:length(C)
        x[i] = -log(x[i])
    end
    sx = sum(x)
    for i in 1:length(C)
        x[i] = ϕ(C,r * x[i]/sx)
    end
    return x
end
ϕ(  C::WilliamsonCopula,      t) = C.ϕ(t)