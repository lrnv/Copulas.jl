
###############################################################################
#####  SklarDist framework.
#####  User-facing function: `SklarDist(C::Copula{d}, m::NTuple{d, <:UnivariateDistribution}) where d`
#####
#####  Nothing here should be overwritten when defining new copulas. 
###############################################################################

"""
    SklarDist(C, margins)

Construct a multivariate distribution from copula `C` and one univariate
distribution per coordinate. The order of `margins` determines which marginal
is attached to each copula coordinate, and its length must equal `length(C)`.

The joint CDF follows Sklar's representation

```math
F(x_1,\\ldots,x_d)=C(F_1(x_1),\\ldots,F_d(x_d)).
```

!!! note "Theorem - Sklar 1959"
    Every multivariate distribution admits such a copula. It is unique on the
    product of the marginal ranges and, in particular, unique when all margins
    are continuous.

`cdf`, `rand`, marginalization, conditioning, and Rosenblatt transforms are
available through the corresponding distribution and Copulas.jl interfaces.
The usual density factorization is available when the copula and every margin
provide the required densities. With discrete or mixed margins, `pdf` and
`logpdf` evaluate the probability mass of Sklar's representation
[genest2007](@cite): a mixed derivative of the copula CDF in the continuous
coordinates and a finite difference over the latent interval
`(F_j(x_j⁻), F_j(x_j)]` in each discrete one, multiplied by the continuous
marginal densities only. That difference
costs `2^k` copula CDF evaluations for `k` discrete margins, which is the
mathematics rather than the implementation; a high-dimensional discrete model
calls for a simulated likelihood. Conditioning and the Rosenblatt transforms
treat a discrete observation as the same latent interval, so `rosenblatt`
draws the atom's distributional transform and is random on atoms.

Passing an [`EmpiricalCopula`](@ref) is allowed for compatibility and emits a
warning because its finite-sample margins are not exactly uniform. The result
therefore does not generally have the requested margins. It represents the
atomic empirical sample transformed by the marginal quantiles, and its `pdf`
and `logpdf` values are generalized point masses rather than Lebesgue
densities. Prefer [`BetaCopula`](@ref), a valid [`BernsteinCopula`](@ref), or
[`CheckerboardCopula`](@ref) when a genuine copula is required.

# Example

```julia
using Copulas, Distributions

C = ClaytonCopula(3, 0.7)
D = SklarDist(C, (Gamma(2, 3), Pareto(), LogNormal()))
sample = rand(D, 1000)
```

For fitting, `SklarDist{CopulaType,Tuple{MarginTypes...}}` is a deliberately
supported public target syntax. It specifies the copula family and one marginal
family per coordinate, for example
`SklarDist{ClaytonCopula,Tuple{Gamma,Normal}}`. Those family parameters are
public in this fitting context; no other field layout, storage parameter, or
concrete representation detail of `SklarDist` is part of the public API. Use
`fit(CopulaModel, SklarDist{...}, data)` to retain the fitted likelihood and
the minimal state needed for diagnostics, inference, and reproducible refitting.

References: 
* [sklar1959](@cite) Sklar, M. (1959). Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP (Vol. 8, No. 3, pp. 229-231).
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
* [genest2007](@cite) Genest, C., & Nešlehová, J. (2007). A primer on copulas for count data. ASTIN Bulletin, 37(2), 475-515.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m) ||
            throw(ArgumentError("every SklarDist margin must be a univariate distribution"))
        if _is_empirical_copula(C)
            @warn "EmpiricalCopula has finite-sample step margins rather than exact uniform margins. The resulting SklarDist therefore does not generally have the requested margins. Consider smoothing with BetaCopula, a valid BernsteinCopula, CheckerboardCopula, or another genuine copula estimator."
        end
        return new{typeof(C),typeof(m)}(C,m)
    end
end
function SklarDist(C::Copula, m)
    margins = Tuple(m)
    length(margins) == length(C) || throw(DimensionMismatch(
        "the number of margins must match the copula dimension",
    ))
    return SklarDist(C, margins)
end
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
function Base.eltype(S::SklarDist)
    T = mapreduce(Distributions.partype, promote_type, S.m; init=Union{})
    return T === Union{} ? Float64 : float(T)
end
function Distributions.partype(S::SklarDist)
    return promote_type(
        Distributions.partype(S.C),
        mapreduce(Distributions.partype, promote_type, S.m; init=Union{}),
    )
end
Distributions.params(S::SklarDist) = (S.C, S.m)
@inline function _sklar_work_eltype(S::SklarDist, x)
    T = promote_type(eltype(S.C), eltype(x))
    for margin in S.m
        T = promote_type(T, Distributions.partype(margin))
    end
    return T
end
function Distributions.cdf(S::SklarDist{CT,TplMargins}, x) where {CT,TplMargins}
    d = length(S)
    length(x) == d || throw(DimensionMismatch(
        "input vector has length $(length(x)); expected distribution dimension $d",
    ))
    T = _sklar_work_eltype(S, x)
    u = Vector{T}(undef, d)
    @inbounds for i in 1:d
        u[i] = Distributions.cdf(S.m[i], x[i])
    end
    return Distributions.cdf(S.C, u)
end
Distributions.logcdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins} = log(Distributions.cdf(S, x))
function Distributions.cdf(S::SklarDist, X::AbstractMatrix)
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.cdf(S, x) for x in eachcol(X)]
end
Distributions.logcdf(S::SklarDist, X::AbstractMatrix) = log.(Distributions.cdf(S, X))
function Distributions.pdf(S::SklarDist, X::AbstractMatrix)
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.pdf(S, x) for x in eachcol(X)]
end
function Distributions.logpdf(S::SklarDist, X::AbstractMatrix)
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.logpdf(S, x) for x in eachcol(X)]
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, A::AbstractMatrix{T}) where {CT,TplMargins,T}
    d = length(S)
    size(A, 1) == d || throw(DimensionMismatch(
        "output matrix has $(size(A, 1)) rows; expected distribution dimension $d",
    ))
    Random.rand!(rng, S.C, A)
    lo, hi = nextfloat(T(0)), prevfloat(T(1))
    @inbounds for col in axes(A, 2), row in axes(A, 1)
        A[row, col] = Distributions.quantile(S.m[row], clamp(A[row, col], lo, hi))
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist, x::AbstractVector{T}) where {T<:Real}
    length(x) == length(S) || throw(DimensionMismatch(
        "output vector has length $(length(x)); expected distribution dimension $(length(S))",
    ))
    Distributions._rand!(rng, S, reshape(x, length(S), 1))
    return x
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins}, u) where {CT,TplMargins}
    d = length(S)
    length(u) == d || throw(DimensionMismatch(
        "input vector has length $(length(u)); expected distribution dimension $d",
    ))
    _has_atoms(S.m) && return _sklar_logpdf_atoms(S, u)
    T = _sklar_work_eltype(S, u)
    # sum marginal logpdfs without generator comprehensions
    s = zero(T)
    @inbounds for i in 1:d
        s += Distributions.logpdf(S.m[i], u[i])
    end
    # compute cdf of marginals, clamped, without broadcasting temporaries
    U = Vector{T}(undef, d)
    @inbounds for i in 1:d
        U[i] = clamp(Distributions.cdf(S.m[i], u[i]), zero(T), one(T))
    end
    return s + Distributions.logpdf(S.C, U)
end
# Sklar's probability mass with atoms: the mixed partial of the copula CDF in
# the continuous coordinates, differenced over the latent interval of each
# discrete coordinate, times the continuous marginal densities.
function _sklar_logpdf_atoms(S::SklarDist, x)
    T = _sklar_work_eltype(S, x)
    s = zero(T)
    cs, ucs = Int[], T[]
    ds, lo, hi = Int[], T[], T[]
    for (i, m) in enumerate(S.m)
        if _is_point_margin(m)
            s += Distributions.logpdf(m, x[i])
            push!(cs, i)
            push!(ucs, clamp(Distributions.cdf(m, x[i]), zero(T), one(T)))
        else
            a, b = _latent_interval(m, x[i])
            push!(ds, i)
            push!(lo, a)
            push!(hi, b)
        end
    end
    mass = _box_partial_cdf(S.C, (), Tuple(cs), Tuple(ds), (), Tuple(ucs), Tuple(lo), Tuple(hi))
    (mass <= 0 || !isfinite(mass)) && return T(-Inf)
    return s + log(mass)
end
function StatsBase.dof(S::SklarDist)
    a = hasmethod(StatsBase.dof, Tuple{typeof(S.C)}) ?
        StatsBase.dof(S.C) : _parameter_dof(Distributions.params(S.C))
    b = sum(hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d)) for d in S.m)
    return a+b
end

_parameter_dof(x::Number) = 1
_parameter_dof(x::Tuple) = sum(_parameter_dof, x; init=0)
_parameter_dof(x::AbstractArray{<:Number}) = length(x)
_parameter_dof(x::Copula) = _parameter_dof(Distributions.params(x))
_parameter_dof(::Any) = 0
