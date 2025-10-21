###########################################################################
#####  Student t Copula (TCopula) fast-paths
###########################################################################
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

function Distributions.logpdf(d::StudentDistortion, u::Real)
    (0 < u < 1) || return -Inf
    σ = d.σz
    σ > 0 || return -Inf
    ν = d.ν
    νp = d.νp
    (ν > 0 && νp > 0) || return -Inf

    Tu = Distributions.TDist(ν)
    Tcond = Distributions.TDist(νp)
    z = Distributions.quantile(Tu, u)
    w = (z - float(d.μz)) / σ

    # log f(u) = log f_Tcond(w) - log σ - log f_T(z)
    return Distributions.logpdf(Tcond, w) - log(σ) - Distributions.logpdf(Tu, z)
end