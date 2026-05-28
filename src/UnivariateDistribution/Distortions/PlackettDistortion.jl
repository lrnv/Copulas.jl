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