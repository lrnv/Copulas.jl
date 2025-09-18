"""
        StudentDistortion(μz, σz, ν, νp)

Parameters
    * `μz` – conditional location shift
    * `σz > 0` – conditional scale
    * `ν` – original t degrees of freedom
    * `νp` – conditional t degrees of freedom (ν + 1 typically)

Conditional distortion for the t copula (elliptical) mapping uniforms through
t quantiles; used by specialized conditioning routines.
"""
struct StudentDistortion{T} <: Distortion
    μz::T
    σz::T
    ν::Int
    νp::Int
end
function Distributions.cdf(d::StudentDistortion, u::Real)
    Tu = Distributions.TDist(d.ν); Tcond = Distributions.TDist(d.νp)
    z = Distributions.quantile(Tu, float(u))
    return Distributions.cdf(Tcond, (z - d.μz) / d.σz)
end
function Distributions.quantile(d::StudentDistortion, α::Real)
    Tu = Distributions.TDist(d.ν); Tcond = Distributions.TDist(d.νp)
    zα = Distributions.quantile(Tcond, float(α))
    return Distributions.cdf(Tu, d.μz + d.σz * zα)
end
## Methods moved next to TCopula type
