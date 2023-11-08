struct ClaytonWilliamsonDistribution{T<:Real,TI} <: Distributions.DiscreteUnivariateDistribution
    θ::T
    d::TI
end
function Distributions.cdf(D::ClaytonWilliamsonDistribution, x::Real)
    θ = D.θ
    d = D.d
    if x < 0
        return zero(x)
    end
    α = -1/θ
    if θ < 0
        if x >= α
            return one(x)
        end
        rez = zero(x)
        x_α = x/α
        for k in 0:(d-1)
            rez += SpecialFunctions.gamma(α+1)/SpecialFunctions.gamma(α-k+1)/SpecialFunctions.gamma(k+1) * (x_α)^k * (1 - x_α)^(α-k)
        end
        return 1-rez
    elseif θ == 0
        return exp(-x)
    else
        rez = zero(x)
        for k in 0:(d-1)
            pr = one(θ)
            for j in 0:(k-1)
                pr *= (1+j*θ)
            end
            rez += pr / SpecialFunctions.gamma(k+1) * x^k * (1 + θ * x)^(-(1/θ+k))
        end
        return 1-rez
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::ClaytonWilliamsonDistribution)
    u = rand(rng)
    Roots.find_zero(x -> (Distributions.cdf(d,x) - u), (0.0, Inf))
end