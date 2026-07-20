###########################################################################
#####  SurvivalCopula + SubsetCopula bindings (delegation)
###########################################################################
struct FlipDistortion{Disto} <: Distortion
    base::Disto
end
Distributions.cdf(D::FlipDistortion, u::Real) = 1 - Distributions.cdf(D.base, 1 - u)
function Distributions.logcdf(D::FlipDistortion, u::Real)
    T = typeof(float(u))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    return LogExpFunctions.log1mexp(Distributions.logcdf(D.base, one(T) - T(u)))
end
Distributions.quantile(D::FlipDistortion, α::Real) = 1 - Distributions.quantile(D.base, 1 - α)

## Methods moved next to SurvivalCopula type
Distributions.logpdf(D::FlipDistortion, u::Real) = Distributions.logpdf(D.base, 1.0 - float(u))
