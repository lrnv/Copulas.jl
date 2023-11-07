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
williamson_dist(C::WilliamsonCopula) = C.X
ϕ(C::WilliamsonCopula, t) = C.ϕ(t)