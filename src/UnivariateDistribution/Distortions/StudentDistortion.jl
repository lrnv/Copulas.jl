###########################################################################
#####  Student t Copula (TCopula) fast-paths
###########################################################################
struct StudentDistortion{T} <: Distortion
    μz::T
    σz::T
    ν::Int
    νp::Int
end
@inline function Distributions.cdf(d::StudentDistortion, u::Real)
    Tu = Distributions.TDist(d.ν); Tcond = Distributions.TDist(d.νp)
    z = Distributions.quantile(Tu, float(u))
    return Distributions.cdf(Tcond, (z - d.μz) / d.σz)
end
@inline function Distributions.quantile(d::StudentDistortion, α::Real)
    Tu = Distributions.TDist(d.ν); Tcond = Distributions.TDist(d.νp)
    zα = Distributions.quantile(Tcond, float(α))
    return Distributions.cdf(Tu, d.μz + d.σz * zα)
end
## Methods moved next to TCopula type
