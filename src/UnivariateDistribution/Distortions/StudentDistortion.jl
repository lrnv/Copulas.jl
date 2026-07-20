###########################################################################
#####  Student t Copula (TCopula) fast-paths
###########################################################################
struct StudentDistortion{T,TU,TC} <: Distortion
    μz::T
    σz::T
    ν::Int
    νp::Int
    Tu::TU
    Tcond::TC
end
function StudentDistortion(μz::Real, σz::Real, ν::Integer, νp::Integer)
    μz, σz = promote(float(μz), float(σz))
    ν, νp = Int(ν), Int(νp)
    Tu = Distributions.TDist(ν)
    Tcond = Distributions.TDist(νp)
    return StudentDistortion{typeof(μz),typeof(Tu),typeof(Tcond)}(
        μz, σz, ν, νp, Tu, Tcond
    )
end
function Distributions.cdf(d::StudentDistortion, u::Real)
    z = Distributions.quantile(d.Tu, float(u))
    return Distributions.cdf(d.Tcond, (z - d.μz) / d.σz)
end
function Distributions.logcdf(d::StudentDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz)))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    z = Distributions.quantile(d.Tu, T(u))
    return T(Distributions.logcdf(d.Tcond, (z - T(d.μz)) / T(d.σz)))
end
function Distributions.quantile(d::StudentDistortion, α::Real)
    zα = Distributions.quantile(d.Tcond, float(α))
    return Distributions.cdf(d.Tu, d.μz + d.σz * zα)
end
## Methods moved next to TCopula type
function Distributions.logpdf(d::StudentDistortion, u::Real)
    z = Distributions.quantile(d.Tu, float(u))
    return Distributions.logpdf(d.Tcond, (z - d.μz) / d.σz) - log(abs(d.σz)) - Distributions.logpdf(d.Tu, z)
end
