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

# Frailty: M = S_{1/δ} * Gamma_{1/θ}^{δ}
struct GammaStoppedPositiveStable{Tα,Tβ} <: Distributions.ContinuousUnivariateDistribution
    α::Tα   # = 1/δ
    β::Tβ   # = 1/θ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedPositiveStable)
    α, β = float(D.α), float(D.β)
    T = rand(rng, Distributions.Gamma(β, 1.0))        # Gamma(shape=β, scale=1)
    return rand(rng, PStable(α; scale = T^(1/α)))     # Z(T)
end

struct GammaStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ
    δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::GammaStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, Distributions.Gamma(1/θ, 1.0))        # shape=1/θ, scale=1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))     # shape=T/δ,  scale=1
end

# --------- Frailty: positive-stable stopped gamma ----------
# T ~ Postable(α = 1/θ, scale = 1);   M | T=t  ~ Gamma(t/δ, 1)
struct PosStableStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ; δ::Tδ
end
function Distributions.rand(rng::Distributions.AbstractRNG, D::PosStableStoppedGamma)
    θ, δ = float(D.θ), float(D.δ)
    T = rand(rng, PStable(inv(θ)))                # Postable(α=1/θ) con escala 1
    return rand(rng, Distributions.Gamma(T/δ, 1.0))
end

struct TiltedPositiveStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T   # 0<α<=1
    λ::T   # λ>=0
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::TiltedPositiveStable)
    α, λ = float(D.α), float(D.λ)
    while true
        x = rand(rng, PStable(α))                  # X ~ Postable(α)
        if rand(rng) ≤ exp(-λ*x)                   # acepta con prob e^{-λ X}
            return x                               # ← M
        end
    end
end