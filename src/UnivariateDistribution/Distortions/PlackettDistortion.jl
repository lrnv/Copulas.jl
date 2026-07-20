###########################################################################
#####  PlackettCopula fast-path (bivariate, p=1)
###########################################################################
struct PlackettDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::PlackettDistortion, u::Real) 
    θ, v = D.θ, D.uⱼ

    η = θ - 1
    t1 = η * (u + v) + 1
    s1 = sqrt(t1 * t1 - 4θ * η * u * v)
    num = log1p(- (t1 - 2θ * u) / s1)

    t2 = η * (1 + v) + 1
    s2 = sqrt(t2 * t2 - 4θ * η * v)
    den = log1p(- (t2 - 2 * θ) / s2)

    return num - den
end
function Distributions.quantile(D::PlackettDistortion, α::Real)
    T = float(promote_type(typeof(α), typeof(D.θ), typeof(D.uⱼ)))
    q = T(α)
    ϵ = eps(T)
    q < ϵ && return zero(T)
    q > one(T) - 2ϵ && return one(T)

    # Closed-form inversion of the conditional Plackett CDF. This is the same
    # inversion used by PlackettCopula sampling, with q as the conditional rank
    # and u the fixed coordinate.
    θ = T(D.θ)
    u = T(D.uⱼ)
    a = q * (one(T) - q)
    b = θ + a * (θ - one(T))^2
    c = 2a * (u * θ^2 + one(T) - u) + θ * (one(T) - 2a)
    radicand = θ + 4a * u * (one(T) - u) * (one(T) - θ)^2
    d = sqrt(θ) * sqrt(max(zero(T), radicand))
    value = (c - (one(T) - 2q) * d) / (2b)
    return clamp(value, zero(T), one(T))
end
## DistortionFromCop moved next to PlackettCopula
function Distributions.logpdf(D::PlackettDistortion, u::Real)
    θ, v = D.θ, D.uⱼ
    η = θ - one(θ)

    # θ == 1 (independence) -> uniform conditional density
    if η == zero(η)
        return zero(eltype(θ))
    end

    t1 = η * (u + v) + one(θ)
    s1 = sqrt(max(zero(θ), t1 * t1 - 4θ * η * u * v))

    # A(u) = 1 - (t1 - 2θ*u)/s1
    # derivative A'(u) = -d/du[(t1 - 2θ*u)/s1]
    dt1 = η
    ds1 = η * (t1 - 2θ * v) / s1
    dB = ((dt1 - 2θ) * s1 - (t1 - 2θ * u) * ds1) / (s1 * s1)
    Ap = -dB

    t2 = η * (1 + v) + one(θ)
    s2 = sqrt(max(zero(θ), t2 * t2 - 4θ * η * v))

    denomA = 1 - (t2 - 2θ) / s2

    return log(abs(Ap)) - log(denomA)
end
