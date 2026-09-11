
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
provide the required densities. With discrete or mixed margins the CDF remains
meaningful, but a product of marginal densities and a copula density does not
in general represent the probability mass; users should not assume `pdf` or
Rosenblatt round-trip properties beyond the capabilities documented by the
components.

# Example

```julia
using Copulas, Distributions

C = ClaytonCopula(3, 0.7)
D = SklarDist(C, (Gamma(2, 3), Pareto(), LogNormal()))
sample = rand(D, 1000)
```

Use `fit(CopulaModel, SklarDist{...}, data)` to estimate specified marginal and
copula families while retaining fitting diagnostics.

References: 
* [sklar1959](@cite) Sklar, M. (1959). Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP (Vol. 8, No. 3, pp. 229-231).
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
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
Distributions.params(S::SklarDist) = (copula=S.C, margins=S.m)
@inline function _sklar_work_eltype(S::SklarDist, x)
    T = promote_type(eltype(S.C), eltype(x))
    for margin in S.m
        T = promote_type(T, Distributions.partype(margin))
    end
    return T
end
function Distributions.cdf(S::SklarDist{CT,TplMargins}, x) where {CT,TplMargins}
    d = length(S)
    length(x) == d || throw(ArgumentError("Dimension mismatch between distribution and input vector"))
    T = _sklar_work_eltype(S, x)
    u = Vector{T}(undef, d)
    @inbounds for i in 1:d
        u[i] = Distributions.cdf(S.m[i], x[i])
    end
    return Distributions.cdf(S.C, u)
end
Distributions.logcdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins} = log(Distributions.cdf(S, x))
function Distributions.cdf(S::SklarDist, X::AbstractMatrix)
    size(X, 1) == length(S) || throw(ArgumentError("Dimension mismatch between distribution and input matrix"))
    return [Distributions.cdf(S, x) for x in eachcol(X)]
end
Distributions.logcdf(S::SklarDist, X::AbstractMatrix) = log.(Distributions.cdf(S, X))
function Distributions.pdf(S::SklarDist, X::AbstractMatrix)
    size(X, 1) == length(S) || throw(ArgumentError("Dimension mismatch between distribution and input matrix"))
    return [Distributions.pdf(S, x) for x in eachcol(X)]
end
function Distributions.logpdf(S::SklarDist, X::AbstractMatrix)
    size(X, 1) == length(S) || throw(ArgumentError("Dimension mismatch between distribution and input matrix"))
    return [Distributions.logpdf(S, x) for x in eachcol(X)]
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, A::AbstractMatrix{T}) where {CT,TplMargins,T}
    size(A, 1) == length(S) || throw(ArgumentError("Dimension mismatch between distribution and output matrix"))
    Random.rand!(rng, S.C, A)
    lo, hi = nextfloat(T(0)), prevfloat(T(1))
    @inbounds for col in axes(A, 2), row in axes(A, 1)
        A[row, col] = Distributions.quantile(S.m[row], clamp(A[row, col], lo, hi))
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist, x::AbstractVector{T}) where {T<:Real}
    Distributions._rand!(rng, S, reshape(x, length(S), 1))
    return x
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins}, u) where {CT,TplMargins}
    d = length(S)
    length(u) == d || throw(ArgumentError("Dimension mismatch between distribution and input vector"))
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
function StatsBase.dof(S::SklarDist)
    a = hasmethod(StatsBase.dof, Tuple{typeof(S.C)}) ?
        StatsBase.dof(S.C) : _parameter_dof(Distributions.params(S.C))
    b = sum(hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d)) for d in S.m)
    return a+b
end

_parameter_dof(x::Number) = 1
_parameter_dof(x::NamedTuple) = sum(_parameter_dof, values(x); init=0)
_parameter_dof(x::Tuple) = sum(_parameter_dof, x; init=0)
_parameter_dof(x::AbstractArray{<:Number}) = length(x)
_parameter_dof(x::Copula) = _parameter_dof(Distributions.params(x))
_parameter_dof(::Any) = 0
