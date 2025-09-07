# Estable positivo (Postable) con escala opcional
struct PStable{Tα<:Real,Ts<:Real} <: Distributions.ContinuousUnivariateDistribution
    α::Tα
    scale::Ts
    function PStable(α::Real; scale::Real=1.0)
        (0.0 < α ≤ 1.0) || throw(ArgumentError("α ∈ (0,1]"))
        new{typeof(α),typeof(scale)}(α,scale)
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::PStable)
    α = float(d.α)
    if α == 1.0
        return d.scale              # S₁ ≡ scale
    else
        V = rand(rng, Distributions.Uniform(0,π))
        W = rand(rng, Distributions.Exponential())
        S = ( sin(α*V) / (sin(V))^(1/α) ) * ( sin((1-α)*V)/W )^((1-α)/α)
        return d.scale * S
    end
end
