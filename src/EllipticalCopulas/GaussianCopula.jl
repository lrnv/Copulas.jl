"""
    GaussianCopula(Σ)
    GaussianCopula(d, Σ)
    GaussianCopula(d, ρ)
    GaussianCopula{d}(Σ)
    GaussianCopula{d}(ρ)

Where `Σ` is a (symmetric) covariance or correlation matrix. The two-argument
form with `(d, ρ)` builds the equicorrelation matrix with ones on the diagonal
and constant off-diagonal correlation `ρ`:

```julia
Σ = fill(ρ, d, d); Σ[diagind(Σ)] .= 1
C = GaussianCopula(d, ρ)            # == GaussianCopula(Σ)
```

Validity domain (equicorrelated PD matrix): `-1/(d-1) < ρ < 1`. The boundary
`ρ = -1/(d-1)` is singular and rejected. If `ρ == 0`, the resulting
`GaussianCopula` represents independence, as it does for any diagonal matrix.

The Gaussian copula is the copula of a multivariate normal distribution. It is defined by

```math
C(\\mathbf{x}; \\boldsymbol{\\Sigma}) = F_{\\Sigma}(F_{\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\Sigma}`` is the cdf of a centered multivariate normal with covariance/correlation ``\\Sigma`` and ``F_{\\Sigma,i}`` its i-th marginal cdf.

Example usage:
```julia
using Copulas, Distributions

Σ = [1.0 0.6; 0.6 1.0]
C = GaussianCopula(Σ)
u = rand(C, 1000)
logpdf(C, u[:, 1]), cdf(C, u[:, 1])
Ĉ = fit(GaussianCopula, u)
```

Special case:
- If `isdiag(Σ)`, the `GaussianCopula` represents independence while retaining
  its concrete family type.

Covariance-like inputs are copied and normalized to correlation scale, and
non-positive-definite matrices are rejected. The copula owns its normalized
matrix; mutating the constructor input or a matrix returned by `params` does not
change the model. Gaussian copulas are asymptotically independent in both tails
for every non-degenerate correlation, and multivariate CDF values are numerical
estimates.

See also: [`TCopula`](@ref), [`SklarDist`](@ref),
[`Nataf`](@ref), [`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GaussianCopula{d,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function GaussianCopula{d}(Σ::AbstractMatrix) where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
        size(Σ) == (d, d) || throw(DimensionMismatch("Σ must be a $d×$d matrix"))
        matrix = Matrix(float.(Σ))
        make_cor!(matrix)
        N(GaussianCopula)(matrix)
        return new{d,typeof(matrix)}(matrix)
    end
end
GaussianCopula(Σ::AbstractMatrix) = GaussianCopula{size(Σ, 1)}(Σ)

# Equicorrelation convenience constructor
function GaussianCopula{d}(ρ::Real) where {d}
    d < 2 && throw(ArgumentError("Use a bivariate or higher dimension (d ≥ 2) or pass a 1×1 matrix."))
    # Positive definiteness condition for equicorrelation matrix
    lower = -1/(d-1)
    ρ ≤ lower && throw(ArgumentError("Equicorrelation value ρ=$(ρ) not in (-1/(d-1), 1). For d=$d the lower open bound is $(lower)."))
    ρ ≥ 1 && throw(ArgumentError("Equicorrelation value ρ must be < 1."))
    Σ = fill(float(ρ), d, d)
    @inbounds for i in 1:d
        Σ[i,i] = one(ρ)
    end
    return GaussianCopula{d}(Σ)
end
GaussianCopula(d::Int, ρ::Real) = GaussianCopula{d}(ρ)
GaussianCopula(d::Int, Σ::AbstractMatrix) = GaussianCopula{d}(Σ)
(::Type{GaussianCopula{D,MT}})(d::Int, Σ::AbstractMatrix) where {D,MT} = GaussianCopula{d}(Σ)
(::Type{GaussianCopula{D,MT}})(d::Int, ρ::Real) where {D,MT} = GaussianCopula{d}(ρ)

U(::Type{T}) where T<: GaussianCopula = Distributions.Normal()
N(::Type{T}) where T<: GaussianCopula = Distributions.MvNormal
function _cdf(C::CT,u) where {CT<:GaussianCopula}
    # MvNormalCDF mutates its upper-bound work vector. HCubature supplies
    # immutable StaticArrays to integrands, so always hand the backend a
    # mutable dense vector.
    x = collect(StatsBase.quantile.(Distributions.Normal(), u))
    d = length(C)
    return MvNormalCDF.mvnormcdf(C.Σ, fill(-Inf, d), x; rng = Random.Xoshiro(0))[1]
end

function rosenblatt(C::GaussianCopula, u::AbstractMatrix{<:Real})
    L = LinearAlgebra.cholesky(C.Σ).L
    z = Distributions.quantile.(Distributions.Normal(), u)
    return Distributions.cdf.(Distributions.Normal(), L \ z)
end

function inverse_rosenblatt(C::GaussianCopula, s::AbstractMatrix{<:Real})
    return Distributions.cdf.(Distributions.Normal(), LinearAlgebra.cholesky(C.Σ).L * Distributions.quantile.(Distributions.Normal(), s))
end

τ(C::GaussianCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π
ρ(C::GaussianCopula{2,MT}) where MT = 6*asin(C.Σ[1,2]/2)/π

function distortion(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real}, i::Int) where {D,MT,p}
    ist = Tuple(setdiff(1:D, js))
    @assert i in ist
    J = collect(js)
    zⱼ = Distributions.quantile.(Distributions.Normal(), collect(uⱼₛ))
    if length(J) == 1
        μz = C.Σ[i, J[1]] * zⱼ[1]
        σz = sqrt(one(μz) - C.Σ[i, J[1]]^2)
    else
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(C.Σ[J, J]))
        β = F \ C.Σ[J, i]
        μz = LinearAlgebra.dot(β, zⱼ)
        σ² = one(μz) - LinearAlgebra.dot(C.Σ[i, J], β)
        σz = sqrt(max(zero(σ²), σ²))
    end
    return GaussianDistortion(float(μz), float(σz))
end
function conditional_copula(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, ::NTuple{p,<:Real}) where {D,MT,p}
    @assert 0 < p < D-1
    J = collect(Int, js)
    I = collect(setdiff(1:D, J))
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(C.Σ[J, J]))
    ΣIJ = C.Σ[I, J]
    Σcond = C.Σ[I, I] - ΣIJ * (F \ C.Σ[J, I])
    return GaussianCopula{D - p}(Σcond)
end

function _conditional_components(C::GaussianCopula{D,MT}, js::NTuple{p,Int},
                                 uⱼₛ::NTuple{p,<:Real}, is) where {D,MT,p}
    J = collect(Int, js)
    I = collect(Int, is)
    Σ = C.Σ
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ[J, J]))
    zJ = Distributions.quantile.(Distributions.Normal(), collect(uⱼₛ))
    ΣIJ = Σ[I, J]
    μ = ΣIJ * (F \ zJ)
    Σcond = Σ[I, I] - ΣIJ * (F \ Σ[J, I])
    distortions = ntuple(k -> begin
        σ² = max(Σcond[k, k], zero(eltype(Σcond)))
        GaussianDistortion(float(μ[k]), float(sqrt(σ²)))
    end, length(is))
    return GaussianCopula{length(is)}(Σcond), distortions
end

SubsetCopula(C::GaussianCopula, dims::NTuple{p, Int}) where p = GaussianCopula{p}(C.Σ[collect(dims),collect(dims)])

StatsBase.dof(C::Copulas.GaussianCopula)    = (p = length(C); p*(p-1) ÷ 2)
Distributions.params(C::GaussianCopula) = (; Σ = copy(C.Σ))
_example(::Type{<:GaussianCopula}, d::Int) = GaussianCopula(d, 0.2)
function _unbound_params(::Type{<:GaussianCopula}, d::Int, θ::NamedTuple)
    return _unbound_corr_params(d, θ.Σ)
end
function _rebound_params(::Type{<:GaussianCopula}, d::Int, α::AbstractVector{T}) where {T}
    return (; Σ = _rebound_corr_params(d, α))
end
function _fit(CT::Type{<:GaussianCopula}, Udata, ::Val{:mle}; weights=nothing)
    d, n = size(Udata)
    N01 = Distributions.Normal()
    Z = Distributions.quantile.(N01, Udata)
    # Cross-product sufficient for the Gaussian copula likelihood. A weighted
    # sample scales each score column by the root of its weight, so that the
    # product stays a symmetric rank-k update, and its size is the weight total.
    Q = weights === nothing ? Z * Z' : (Zw = Z .* sqrt.(weights)'; Zw * Zw')
    n = weights === nothing ? n : sum(weights)
    if d == 2
        q11 = Q[1, 1]; q22 = Q[2, 2]; q12 = Q[1, 2]
        T = eltype(Q)
        δ = sqrt(eps(T))
        lower = -one(T) + δ
        upper =  one(T) - δ
        objective_2d = ρ -> begin
            one_minus_ρ² = one(ρ) - ρ * ρ
            return n / 2 * log(one_minus_ρ²) + (q11 + q22 - 2ρ * q12) / (2 * one_minus_ρ²)
        end
        res = Optim.optimize(objective_2d, lower, upper, Optim.Brent(),)
        ρ̂ = Optim.minimizer(res)
        R̂ = T[one(T) ρ̂; ρ̂ one(T)]
        return GaussianCopula(R̂)
    end
    R₀ = _score_corr_start(Z)
    α₀ = _unbound_corr_params(d, R₀)
    objective_hd = α -> begin
        L = _rebound_corr_factor(d, α)
        Ltri = LinearAlgebra.LowerTriangular(L)
        logdetR = 2 * sum(log, LinearAlgebra.diag(L))
        Y = Ltri \ Q
        RinvQ = transpose(Ltri) \ Y
        quadratic = LinearAlgebra.tr(RinvQ)
        return (n * logdetR + quadratic) / 2
    end
    res = Optim.optimize(
        objective_hd,
        α₀,
        Optim.LBFGS();
        autodiff=ADTypes.AutoForwardDiff(),
    )
    α̂ = Optim.minimizer(res)
    L̂ = _rebound_corr_factor(d, α̂)
    R̂ = L̂ * L̂'
    R̂ = (R̂ + R̂') / 2
    return GaussianCopula(R̂)
end
function _fit(::Type{<:GaussianCopula}, U, ::Val{:itau})
    τ̂ = StatsBase.corkendall(U')
    return GaussianCopula(_nearest_correlation(sinpi.(τ̂ ./ 2)))
end
function _fit(::Type{<:GaussianCopula}, U, ::Val{:irho})
    ρ̂ = StatsBase.corspearman(U')
    return GaussianCopula(_nearest_correlation(2 .* sinpi.(ρ̂ ./ 6)))
end
_available_fitting_methods(::Type{<:GaussianCopula}, d) = (:mle, :itau, :irho, :ibeta)
