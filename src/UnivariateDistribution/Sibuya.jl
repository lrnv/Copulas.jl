# See https://rdrr.io/rforge/copula/man/Sibuya.html

struct Sibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    p::T
    function Sibuya(p::T) where {T <: Real}
        new{T}(p)
    end
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::Sibuya{T}) where {T <: Real}
    u = rand(rng, T)
    if u <= d.p
        return T(1)
    end
    xMax = 1/eps(T)
    Ginv = ((1-u)*SpecialFunctions.gamma(1-d.p))^(-1/d.p)
    fGinv = floor(Ginv)
    if Ginv > xMax 
        return fGinv
    end
    if 1-u < 1/(fGinv*SpecialFunctions.beta(fGinv,1-d.p))
        return ceil(Ginv)
    end
    return fGinv
end

# -------- Frailty: Sibuya-stopped-Positive-Stable --------
struct SibuyaStoppedPosStable{T} <: Distributions.ContinuousUnivariateDistribution
    θ::T   # ≥ 1
    δ::T   # ≥ 1
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::SibuyaStoppedPosStable)
    θ = float(D.θ); δ = float(D.δ)
    t = rand(rng, Sibuya(1/θ))                 # T ~ Sibuya(1/θ)
    return rand(rng, PStable(1/δ; scale = t^δ))# M | T=t ~ Postable(α=1/δ, scale=t^δ)
end

# -------- Frailty: Sibuya-stopped-Gamma --------
struct SibuyaStoppedGamma{Tθ,Tδ} <: Distributions.ContinuousUnivariateDistribution
    θ::Tθ   # ≥ 1
    δ::Tδ   # > 0
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::SibuyaStoppedGamma)
    θ = float(D.θ); δ = float(D.δ)
    # T ~ Sibuya(1/θ) 
    t = rand(rng, Sibuya(1/θ))
    # M | T=t ~ Gamma(t/δ, 1)
    return rand(rng, Distributions.Gamma(t/δ, 1.0))
end

# ---------- Frailty: Generalized Sibuya ----------
struct GeneralizedSibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    ϑ::T
    δ::T
    function GeneralizedSibuya(ϑ, δ)
        (ϑ ≥ 1)  || throw(ArgumentError("ϑ must be ≥ 1"))
        (0 < δ ≤ 1) || throw(ArgumentError("δ must be in (0,1]"))
        new{typeof(ϑ)}(ϑ, δ)
    end
end

function Distributions.rand(rng::Distributions.AbstractRNG, d::GeneralizedSibuya)
    ϑ = float(d.ϑ); δ = float(d.δ)
    η = 1 - (1 - δ)^ϑ
    if δ ≤ 1 - exp(-1)
        logser = Logarithmic(log1p(-η))
        while true
            X = rand(rng, logser)
            acc = 1 / ((X - 1/ϑ) * SpecialFunctions.beta(X, 1 - 1/ϑ))
            if rand(rng, Distributions.Uniform(0,1)) ≤ acc
                return X
            end
        end
    else                   
        α = 1/ϑ            
        while true
            X = rand(rng, Sibuya(α))
            if rand(rng, Distributions.Uniform(0,1)) ≤ η^(X-1) 
                return X
            end
        end
    end
end

# ---------- Frailty: Generalized Sibuya ----------
struct ShiftedNegBinFrailty{T} <: Distributions.DiscreteUnivariateDistribution
    r::T   # r = 1/θ
    p::T   # p = 1-π ∈ [0,1]
end
Distributions.rand(rng::Distributions.AbstractRNG, D::ShiftedNegBinFrailty) =
    D.r + rand(rng, Distributions.NegativeBinomial(D.r, D.p))
