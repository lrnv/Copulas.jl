###########################################################################
#####  Student t Copula (TCopula) fast-paths
###########################################################################
struct StudentDistortion{T,Tν} <: Distortion
    μz::T
    σz::T
    ν::Tν
    νp::Tν
end
function StudentDistortion(μz::Real, σz::Real, ν::Real, νp::Real)
    μz, σz = promote(float(μz), float(σz))
    ν, νp = promote(float(ν), float(νp))
    return StudentDistortion{typeof(μz),typeof(ν)}(μz, σz, ν, νp)
end

# Rmath's scalar kernels are substantially faster for the hardware floating-point
# types used by vine workloads. Keep StatsFuns as the generic numeric fallback.
@inline _student_cdf(ν::Float64, x::Float64) = Rmath.pt(x, ν)
@inline _student_cdf(ν::Float32, x::Float32) = Float32(Rmath.pt(x, ν))
@inline _student_cdf(ν::Real, x::Real) = StatsFuns.tdistcdf(ν, x)
@inline _student_logcdf(ν::Float64, x::Float64) = Rmath.pt(x, ν, true, true)
@inline _student_logcdf(ν::Float32, x::Float32) = Float32(Rmath.pt(x, ν, true, true))
@inline _student_logcdf(ν::Real, x::Real) = StatsFuns.tdistlogcdf(ν, x)
@inline _student_quantile(ν::Float64, p::Float64) = Rmath.qt(p, ν)
@inline _student_quantile(ν::Float32, p::Float32) = Float32(Rmath.qt(p, ν))
@inline _student_quantile(ν::Real, p::Real) = StatsFuns.tdistinvcdf(ν, p)
@inline _student_logpdf(ν::Real, x::Real) = StatsFuns.tdistlogpdf(ν, x)

function Distributions.cdf(d::StudentDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz), typeof(d.ν)))
    u <= 0 && return zero(T)
    u >= 1 && return one(T)
    z = _student_quantile(T(d.ν), T(u))
    return _student_cdf(T(d.νp), (z - T(d.μz)) / T(d.σz))
end
function Distributions.logcdf(d::StudentDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz), typeof(d.ν)))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    z = _student_quantile(T(d.ν), T(u))
    return _student_logcdf(T(d.νp), (z - T(d.μz)) / T(d.σz))
end
function Distributions.quantile(d::StudentDistortion, α::Real)
    T = float(promote_type(typeof(α), typeof(d.μz), typeof(d.σz), typeof(d.ν)))
    zα = _student_quantile(T(d.νp), T(α))
    return _student_cdf(T(d.ν), T(d.μz) + T(d.σz) * zα)
end
## Methods moved next to TCopula type
function Distributions.logpdf(d::StudentDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz), typeof(d.ν)))
    0 < u < 1 || return T(-Inf)
    z = _student_quantile(T(d.ν), T(u))
    return _student_logpdf(T(d.νp), (z - T(d.μz)) / T(d.σz)) -
           log(abs(T(d.σz))) - _student_logpdf(T(d.ν), z)
end
