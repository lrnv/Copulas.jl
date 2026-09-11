"""
    TCopula(df, Σ)
    TCopula(d, df, Σ)
    TCopula{d}(df, Σ)

The Student t copula is the copula of a multivariate Student t distribution. It is defined by

```math
C(\\mathbf{x}; \\nu, \\boldsymbol{\\Sigma}) = F_{\\nu,\\Sigma}(F_{\\nu,\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\nu,\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\nu,\\Sigma}`` is the cdf of a centered multivariate t with correlation ``\\Sigma`` and ``\\nu`` degrees of freedom.

The multivariate CDF is evaluated through the normal scale-mixture
representation of the Student distribution. The remaining one-dimensional
radial integral uses adaptive quadrature and the inner normal probabilities use
`MvNormalCDF.jl`; consequently, returned probabilities are numerical estimates.
For bivariate copulas, Spearman's rho uses the one-dimensional formula of
Heinen and Valdesogo (2020).

Example usage:
```julia
using Copulas, Distributions

Σ = [1.0 0.6; 0.6 1.0]
C = TCopula(4.0, Σ)
u = rand(C, 1000)
logpdf(C, u[:, 1]), cdf(C, u[:, 1])
Ĉ = fit(TCopula{2}, u; vcov=false)
```

Degrees of freedom must be positive and the correlation matrix positive
definite. The matrix is normalized to correlation scale in place; pass a copy
when the input must be preserved. Unlike the Gaussian copula, finite degrees
of freedom produce symmetric lower- and upper-tail dependence. Large `ν`
approaches the Gaussian copula and can be weakly identified.

See also: [`EllipticalCopula`](@ref), [`GaussianCopula`](@ref),
[`SklarDist`](@ref), [`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
* [heinen2020spearman](@cite) Heinen, Andréas and Valdesogo, Alfonso.
  Spearman rank correlation of the bivariate Student t and scale mixtures of
  normal distributions. Journal of Multivariate Analysis, 2020.
* [genz1992normal](@cite) Genz, Alan. Numerical computation of multivariate
  normal probabilities. Journal of Computational and Graphical Statistics, 1992.
"""
struct TCopula{d,Tν,MT} <: EllipticalCopula{d,MT}
    df::Tν
    Σ::MT
    function TCopula{d}(df::Real, Σ::AbstractMatrix) where {d}
        size(Σ) == (d, d) || throw(DimensionMismatch("Σ must be a $d×$d matrix"))
        make_cor!(Σ)
        Distributions.MvTDist(df, Σ)
        return new{d,typeof(df),typeof(Σ)}(df, Σ)
    end
end
Base.eltype(C::TCopula) = promote_type(typeof(float(C.df)), eltype(C.Σ))
TCopula(ν::Real, Σ::AbstractMatrix) = TCopula{size(Σ, 1)}(ν, Σ)
TCopula(d::Int, ν::Real, Σ::AbstractMatrix) = TCopula{d}(ν, Σ)
(::Type{TCopula{D,Tν,MT}})(d::Int, ν::Real, Σ::AbstractMatrix) where {D,Tν,MT} = TCopula{d}(ν, Σ)



U(C::TCopula) = Distributions.TDist(C.df)
N(C::TCopula) = function(Σ)
    Distributions.MvTDist(C.df, Σ)
end

function _cdf(C::TCopula{d}, u) where d
    T = promote_type(eltype(C), eltype(u))
    T <: Union{Float32,Float64} ||
        return invoke(_cdf, Tuple{Copula,Any}, C, u)
    upper = Distributions.quantile.(Distributions.TDist(C.df), u)
    return T(_mvtcdf(C.df, zeros(eltype(upper), d), C.Σ, upper))
end

function _student_rosenblatt_cache(C::TCopula{d}) where d
    Σ = C.Σ
    return ntuple(d) do k
        k == 1 && return nothing
        J = 1:(k - 1)
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ[J, J]))
        β = F \ Σ[J, k]
        σ0² = max(Σ[k, k] - LinearAlgebra.dot(Σ[k, J], β), zero(eltype(Σ)))
        return (; F, β, σ0 = sqrt(σ0²))
    end
end

function rosenblatt(C::TCopula{d}, u::AbstractMatrix{<:Real}) where {d}
    size(u, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    ν = C.df
    Tu = Distributions.TDist(ν)
    z = Distributions.quantile.(Tu, u)
    v = similar(z)
    v[1, :] .= u[1, :]
    cache = _student_rosenblatt_cache(C)
    @inbounds for k in 2:d
        entry = cache[k]
        Tcond = Distributions.TDist(ν + k - 1)
        for col in axes(u, 2)
            zJ = view(z, 1:(k - 1), col)
            solved_zJ = entry.F \ zJ
            μ = LinearAlgebra.dot(entry.β, zJ)
            δ = LinearAlgebra.dot(zJ, solved_zJ)
            σ = entry.σ0 * sqrt((ν + δ) / (ν + k - 1))
            v[k, col] = Distributions.cdf(Tcond, (z[k, col] - μ) / σ)
        end
    end
    return v
end

function inverse_rosenblatt(C::TCopula{d}, s::AbstractMatrix{<:Real}) where {d}
    size(s, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    ν = C.df
    Tu = Distributions.TDist(ν)
    z = similar(s, float(promote_type(eltype(s), eltype(C.Σ))))
    v = similar(z)
    z[1, :] .= Distributions.quantile.(Tu, s[1, :])
    v[1, :] .= s[1, :]
    cache = _student_rosenblatt_cache(C)
    @inbounds for k in 2:d
        entry = cache[k]
        Tcond = Distributions.TDist(ν + k - 1)
        for col in axes(s, 2)
            zJ = view(z, 1:(k - 1), col)
            solved_zJ = entry.F \ zJ
            μ = LinearAlgebra.dot(entry.β, zJ)
            δ = LinearAlgebra.dot(zJ, solved_zJ)
            σ = entry.σ0 * sqrt((ν + δ) / (ν + k - 1))
            z[k, col] = μ + σ * Distributions.quantile(Tcond, s[k, col])
            v[k, col] = Distributions.cdf(Tu, z[k, col])
        end
    end
    return v
end

# Kendall tau of bivariate student:
# Lindskog, F., McNeil, A., & Schmock, U. (2003). Kendall’s tau for elliptical distributions. In Credit risk: Measurement, evaluation and management (pp. 149-156). Heidelberg: Physica-Verlag HD.
τ(C::TCopula{2}) = 2*asin(C.Σ[1,2])/π

# Heinen and Valdesogo (2020), Theorem 2. The one-dimensional expression
# avoids repeatedly integrating the bivariate copula CDF.
function ρ(C::TCopula{2})
    ν = float(C.df)
    r = float(C.Σ[1, 2])
    iszero(r) && return zero(promote_type(typeof(ν), typeof(r)))
    isinf(ν) && return 6asin(r / 2) / π
    if ν > 10
        # The zero-balanced hypergeometric term becomes poorly scaled in
        # hardware precision as ν grows. The equivalent density moment remains
        # stable and is still much cheaper than integrating the numerical CDF.
        return 12 * HCubature.hcubature(
            u -> prod(u) * Distributions.pdf(C, u), zeros(2), ones(2);
            rtol=1e-6,
        )[1] - 3
    end

    logconstant = log(2) + 2 * SpecialFunctions.loggamma(ν) +
                  SpecialFunctions.loggamma(3ν / 2) -
                  3 * SpecialFunctions.loggamma(ν / 2) -
                  SpecialFunctions.loggamma(2ν)
    constant = exp(logconstant)
    integrand(v) = begin
        iszero(v) && return zero(v)
        h = HypergeometricFunctions.pFq((ν, ν), (2ν,), 1 - v^2)
        asin(r * v) * constant * v^(ν - 1) * (1 - v^2)^(ν / 2 - 1) * h
    end
    value = QuadGK.quadgk(integrand, zero(ν), one(ν); rtol=1e-8)[1]
    return 6value / π
end

# Conditioning colocated
function distortion(C::TCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p,D}
    ν = C.df
    Σ = C.Σ; jst = js; ist = Tuple(setdiff(1:D, jst)); @assert i in ist
    Jv = collect(jst); zJ = Distributions.quantile.(Distributions.TDist(ν), collect(uⱼₛ))
    ΣJJ = Σ[Jv, Jv]; RiJ = Σ[i, Jv]; RJi = Σ[Jv, i]
    if length(Jv) == 1
        r = RiJ[1]; μz = r * zJ[1]; σ0² = 1 - r^2; δ = zJ[1]^2
    else
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(ΣJJ))
        solved_zJ = F \ zJ
        μz = LinearAlgebra.dot(RiJ, solved_zJ)
        σ0² = 1 - LinearAlgebra.dot(RiJ, F \ RJi)
        δ = LinearAlgebra.dot(zJ, solved_zJ)
    end
    νp = ν + length(Jv); σz = sqrt(max(σ0², zero(σ0²))) * sqrt((ν + δ) / νp)
    return StudentDistortion(float(μz), float(σz), ν, νp)
end
function conditional_copula(C::TCopula{D}, js, uⱼₛ) where {D}
    df = C.df
    p = length(js); J = collect(Int, js); I = collect(setdiff(1:D, J)); Σ = C.Σ
    if p == 1
        Σcond = Σ[I, I] - Σ[I, J] * (Σ[J, J] \ Σ[J, I])
    else
        L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ[J, J]))
        Σcond = Σ[I, I] - Σ[I, J] * (L' \ (L \ Σ[J, I]))
    end
    σ = sqrt.(LinearAlgebra.diag(Σcond))
    R_cond = Matrix(Σcond ./ (σ * σ'))
    return TCopula{D - p}(df + p, R_cond)
end

function _conditional_components(C::TCopula{D}, js::NTuple{p,Int},
                                 uⱼₛ::NTuple{p,Float64}, is) where {D,p}
    ν = C.df
    J = collect(Int, js)
    I = collect(Int, is)
    Σ = C.Σ
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ[J, J]))
    zJ = Distributions.quantile.(Distributions.TDist(ν), collect(uⱼₛ))
    solved_zJ = F \ zJ
    ΣIJ = Σ[I, J]
    μ = ΣIJ * solved_zJ
    Σcond = Σ[I, I] - ΣIJ * (F \ Σ[J, I])
    δ = LinearAlgebra.dot(zJ, solved_zJ)
    νp = ν + p
    scale = sqrt((ν + δ) / νp)
    distortions = ntuple(k -> begin
        σ² = max(Σcond[k, k], zero(eltype(Σcond)))
        StudentDistortion(float(μ[k]), float(sqrt(σ²) * scale), ν, νp)
    end, length(is))
    σ = sqrt.(LinearAlgebra.diag(Σcond))
    Rcond = Matrix(Σcond ./ (σ * σ'))
    return TCopula{length(is)}(νp, Rcond), distortions
end
# Subsetting colocated
SubsetCopula(C::TCopula, dims::NTuple{p, Int}) where {p} = TCopula{p}(C.df, C.Σ[collect(dims),collect(dims)])

# Fitting collocated
StatsBase.dof(C::Copulas.TCopula)           = (p = length(C); p*(p-1) ÷ 2 + 1)
Distributions.params(C::TCopula) = (; ν = C.df, Σ = C.Σ)
_example(::Type{<:TCopula}, d::Int) = TCopula(5.0, Matrix(LinearAlgebra.I, d, d) .+ 0.2 .* (ones(d, d) .- Matrix(LinearAlgebra.I, d, d)))
function _unbound_params(::Type{<:TCopula}, d::Int, θ::NamedTuple)
    α = _unbound_corr_params(d, θ.Σ)
    return vcat(log(θ.ν), α)
end
function _rebound_params(::Type{<:TCopula}, d::Int, α::AbstractVector{T}) where {T}
    ν = exp(α[1])
    Σ = _rebound_corr_params(d, @view α[2:end])
    return (; ν = ν, Σ = Σ)
end


function _fit(::Type{<:TCopula}, U, ::Val{:itau_irho})
    size(U, 1) == 2 || throw(ArgumentError("Student rank matching is only defined in dimension 2"))
    τ̂ = StatsBase.corkendall(U')[1, 2]
    ρ̂ = StatsBase.corspearman(U')[1, 2]
    r = clamp(sinpi(τ̂ / 2), -1 + eps(Float64), 1 - eps(Float64))
    iszero(r) && throw(ArgumentError(
        "Student degrees of freedom are not identifiable from rank correlations when Kendall's tau is zero",
    ))

    target = abs(ρ̂)
    objective(logν) = abs(ρ(TCopula{2}(exp(logν), [1.0 r; r 1.0]))) - target
    lower, middle, upper = log(0.1), log(10.0), log(100.0)
    flo, fmid = objective(lower), objective(middle)
    logν = if flo >= 0
        lower
    elseif fmid >= 0
        Roots.find_zero(objective, (lower, middle), Roots.Bisection())
    else
        gaussian_limit = abs(6asin(r / 2) / π) - target
        if gaussian_limit <= 0
            Inf
        else
            fhi = objective(upper)
            fhi < 0 ? Inf :
                Roots.find_zero(objective, (middle, upper), Roots.Bisection())
        end
    end
    ν = isinf(logν) ? Inf : exp(logν)
    C = TCopula{2}(ν, [1.0 r; r 1.0])
    return C, (; θ̂=(; ν, Σ=C.Σ), τ̂, ρ̂)
end

_available_fitting_methods(::Type{<:TCopula}, d) = d == 2 ? (:mle, :itau_irho) : (:mle,)
