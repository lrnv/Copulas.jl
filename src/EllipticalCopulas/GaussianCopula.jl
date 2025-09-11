"""
    GaussianCopula{d, MT}

Fields:
- `Σ::MT` — correlation matrix (the constructor coerces the input to a correlation matrix).

Constructor

    GaussianCopula(Σ)

The Gaussian copula is the copula of a multivariate normal distribution. It is defined by

```math
C(\\mathbf{x}; \\boldsymbol{\\Sigma}) = F_{\\Sigma}(F_{\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\Sigma}`` is the cdf of a centered multivariate normal with covariance/correlation ``\\Sigma`` and ``F_{\\Sigma,i}`` its i-th marginal cdf.

Example usage:
```julia
C = GaussianCopula(Σ)
u = rand(C, 1000)
pdf(C, u); cdf(C, u)
Ĉ = fit(GaussianCopula, u)
```

Special case:
- If `isdiag(Σ)`, the constructor returns `IndependentCopula(d)`.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GaussianCopula{d,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function GaussianCopula(Σ)
        if LinearAlgebra.isdiag(Σ)
            return IndependentCopula(size(Σ,1))
        end
        make_cor!(Σ)
        N(GaussianCopula)(Σ)
        return new{size(Σ,1),typeof(Σ)}(Σ)
    end
end
U(::Type{T}) where T<: GaussianCopula = Distributions.Normal()
N(::Type{T}) where T<: GaussianCopula = Distributions.MvNormal
function Distributions.fit(::Type{CT},u) where {CT<:GaussianCopula}
    dd = Distributions.fit(N(CT), StatsBase.quantile.(U(CT),u))
    Σ = Matrix(dd.Σ)
    return GaussianCopula(Σ)
end

function _cdf(C::CT,u) where {CT<:GaussianCopula}
    x = StatsBase.quantile.(Distributions.Normal(),u)
    d = length(C)
    T = eltype(u)
    μ = zeros(T,d)
    lb = repeat([T(-Inf)],d)
    return MvNormalCDF.mvnormcdf(μ, C.Σ, lb, x)[1]
end

function rosenblatt(C::GaussianCopula, u::AbstractMatrix{<:Real})
    return Distributions.cdf.(Distributions.Normal(), inv(LinearAlgebra.cholesky(C.Σ).L) * Distributions.quantile.(Distributions.Normal(), u))
end

function inverse_rosenblatt(C::GaussianCopula, s::AbstractMatrix{<:Real})
    return Distributions.cdf.(Distributions.Normal(), LinearAlgebra.cholesky(C.Σ).L * Distributions.quantile.(Distributions.Normal(), s))
end

# Kendall tau of bivariate gaussian:
# Theorem 3.1 in Fang, Fang, & Kotz, The Meta-elliptical Distributions with Given Marginals Journal of Multivariate Analysis, Elsevier, 2002, 82, 1–16
τ(C::GaussianCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π
ρ(C::GaussianCopula{2,MT}) where MT = 6*asin(C.Σ[1,2]/2)/π

# Conditioning and subsetting fast paths colocated with the type
@inline function DistortionFromCop(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,MT,p}
    ist = Tuple(setdiff(1:D, js))
    @assert i in ist
    J = collect(js)
    zⱼ = Distributions.quantile.(Distributions.Normal(), collect(uⱼₛ))
    if length(J) == 1 # if we condition on only one variable
        μz = C.Σ[i, J[1]] * zⱼ[1]
        σz = sqrt(1 - C.Σ[i, J[1]]^2)
    else
        Reg = C.Σ[i:i, J] * inv(C.Σ[J, J])
        μz = (Reg * zⱼ)[1]
        σz = sqrt(1 - (Reg * C.Σ[J, i:i])[1])
    end
    return GaussianDistortion(float(μz), float(σz))
end
@inline function ConditionalCopula(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D,MT,p}
    @assert 0 < p < D-1
    J = collect(Int, js)
    I = collect(setdiff(1:D, J))
    Σcond = C.Σ[I, I] - C.Σ[I, J] * inv(C.Σ[J, J]) * C.Σ[J, I]
    return GaussianCopula(Σcond)
end

# Subsetting colocated
SubsetCopula(C::GaussianCopula, dims::NTuple{p, Int}) where p = GaussianCopula(C.Σ[collect(dims),collect(dims)])

