###########################################################################
#####  SurvivalCopula + SubsetCopula bindings (delegation)
###########################################################################
struct FlipDistortion{Disto} <: Distortion
    base::Disto
    flipped::Bool
end
FlipDistortion(base) = FlipDistortion(base, true)
distortion_measure_style(D::FlipDistortion) = distortion_measure_style(D.base)
function Distributions.cdf(D::FlipDistortion, u::Real)
    D.flipped || return Distributions.cdf(D.base, u)
    u <= 0 && return zero(float(u))
    u >= 1 && return one(float(u))
    return 1 - Distributions.cdf(D.base, 1 - u)
end
function Distributions.logcdf(D::FlipDistortion, u::Real)
    D.flipped || return Distributions.logcdf(D.base, u)
    T = typeof(float(u))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    return LogExpFunctions.log1mexp(Distributions.logcdf(D.base, one(T) - T(u)))
end
Distributions.quantile(D::FlipDistortion, α::Real) = D.flipped ?
    1 - Distributions.quantile(D.base, 1 - α) : Distributions.quantile(D.base, α)

## Methods moved next to SurvivalCopula type
function Distributions.logpdf(D::FlipDistortion, u::Real)
    D.flipped || return Distributions.logpdf(D.base, u)
    0 <= u <= 1 || return typeof(float(u))(-Inf)
    return Distributions.logpdf(D.base, 1.0 - float(u))
end
