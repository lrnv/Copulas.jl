"""
    TCopula(ν, Σ)
    TCopula(d, ν, Σ)
    TCopula{d}(ν, Σ)

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
Ĉ = fit(TCopula{2}, u)
```

Degrees of freedom must be positive. Covariance-like matrix inputs are copied
and normalized to correlation scale, and non-positive-definite matrices are
rejected. The copula owns its normalized matrix; mutating the constructor input
or a matrix returned by `params` does not change the model. Unlike the Gaussian
copula, finite degrees of freedom produce symmetric lower- and upper-tail
dependence. Large `ν` approaches the Gaussian copula and can be weakly
identified.

See also: [`GaussianCopula`](@ref), [`SklarDist`](@ref),
[`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
* [heinen2020spearman](@cite) Heinen, Andréas and Valdesogo, Alfonso.
  Spearman rank correlation of the bivariate Student t and scale mixtures of
  normal distributions. Journal of Multivariate Analysis, 2020.
* [genz1992normal](@cite) Genz, Alan. Numerical computation of multivariate
  normal probabilities. Journal of Computational and Graphical Statistics, 1992.
"""
struct TCopula{d,Tν,MT} <: EllipticalCopula{d,MT}
    ν::Tν
    Σ::MT
    function TCopula{d}(ν::Real, Σ::AbstractMatrix) where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
        size(Σ) == (d, d) || throw(DimensionMismatch("Σ must be a $d×$d matrix"))
        matrix = Matrix(float.(Σ))
        make_cor!(matrix)
        Distributions.MvTDist(ν, matrix)
        return new{d,typeof(ν),typeof(matrix)}(ν, matrix)
    end
end
Base.eltype(C::TCopula) = promote_type(typeof(float(C.ν)), eltype(C.Σ))
TCopula(ν::Real, Σ::AbstractMatrix) = TCopula{size(Σ, 1)}(ν, Σ)
TCopula(d::Int, ν::Real, Σ::AbstractMatrix) = TCopula{d}(ν, Σ)
(::Type{TCopula{D,Tν,MT}})(d::Int, ν::Real, Σ::AbstractMatrix) where {D,Tν,MT} = TCopula{d}(ν, Σ)

U(C::TCopula) = isinf(C.ν) ? Distributions.Normal() : Distributions.TDist(C.ν)
N(C::TCopula) = isinf(C.ν) ? Distributions.MvNormal : (Σ -> Distributions.MvTDist(C.ν, Σ))
@inline _gaussian_limit(C::TCopula) = GaussianCopula(copy(C.Σ))
function _cdf(C::TCopula{d}, u) where d
    isinf(C.ν) && return _cdf(_gaussian_limit(C), u)
    T = promote_type(eltype(C), eltype(u))
    T <: Union{Float32,Float64} || return invoke(_cdf, Tuple{Copula,Any}, C, u)
    upper = Distributions.quantile.(Distributions.TDist(C.ν), u)
    return T(_mvtcdf(C.ν, zeros(eltype(upper), d), C.Σ, upper))
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
    isinf(C.ν) && return rosenblatt(_gaussian_limit(C), u)
    size(u, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    ν = C.ν
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
    isinf(C.ν) && return inverse_rosenblatt(_gaussian_limit(C), s)
    size(s, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    ν = C.ν
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

τ(C::TCopula{2}) = 2*asin(C.Σ[1,2])/π

function ρ(C::TCopula{2})
    ν = float(C.ν)
    r = float(C.Σ[1, 2])
    iszero(r) && return zero(promote_type(typeof(ν), typeof(r)))
    isinf(ν) && return 6asin(r / 2) / π
    if ν > 10
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

function distortion(C::TCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real}, i::Int) where {p,D}
    isinf(C.ν) && return distortion(_gaussian_limit(C), js, uⱼₛ, i)
    ν = C.ν
    Σ = C.Σ; jst = js; ist = Tuple(setdiff(1:D, jst)); @assert i in ist
    Jv = collect(jst); zJ = Distributions.quantile.(Distributions.TDist(ν), collect(uⱼₛ))
    ΣJJ = Σ[Jv, Jv]; RiJ = Σ[i, Jv]; RJi = Σ[Jv, i]
    if length(Jv) == 1
        r = RiJ[1]; μz = r * zJ[1]; σ0² = one(r) - r^2; δ = zJ[1]^2
    else
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(ΣJJ))
        solved_zJ = F \ zJ
        μz = LinearAlgebra.dot(RiJ, solved_zJ)
        σ0² = one(μz) - LinearAlgebra.dot(RiJ, F \ RJi)
        δ = LinearAlgebra.dot(zJ, solved_zJ)
    end
    νp = ν + length(Jv); σz = sqrt(max(σ0², zero(σ0²))) * sqrt((ν + δ) / νp)
    return StudentDistortion(float(μz), float(σz), ν, νp)
end
function conditional_copula(C::TCopula{D}, js, uⱼₛ) where {D}
    ν = C.ν
    p = length(js); J = collect(Int, js); I = collect(setdiff(1:D, J)); Σ = C.Σ
    if p == 1
        Σcond = Σ[I, I] - Σ[I, J] * (Σ[J, J] \ Σ[J, I])
    else
        L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ[J, J]))
        Σcond = Σ[I, I] - Σ[I, J] * (L' \ (L \ Σ[J, I]))
    end
    σ = sqrt.(LinearAlgebra.diag(Σcond))
    R_cond = Matrix(Σcond ./ (σ * σ'))
    return TCopula{D - p}(ν + p, R_cond)
end

function _conditional_components(C::TCopula{D}, js::NTuple{p,Int},
                                 uⱼₛ::NTuple{p,<:Real}, is) where {D,p}
    if isinf(C.ν)
        Gcond, distortions = _conditional_components(_gaussian_limit(C), js, uⱼₛ, is)
        return TCopula(Inf, copy(Gcond.Σ)), distortions
    end
    ν = C.ν
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
SubsetCopula(C::TCopula, dims::NTuple{p, Int}) where {p} = TCopula{p}(C.ν, C.Σ[collect(dims),collect(dims)])

Paramorph.param_space(::Type{<:TCopula}, d) = (Paramorph.Pos(:ν), Paramorph.Correlation(:Σ, d))
# Per-observation sums of the Student objective. The weighted form multiplies
# the term of each column by its weight before the same reduction, so unit
# weights reproduce the unweighted sum bit for bit.
_t_weighted_sum(v::AbstractVector, ::Nothing) = sum(v)
_t_weighted_sum(v::AbstractVector, w::AbstractVector) = sum(w .* v)
_t_weighted_sum(A::AbstractMatrix, ::Nothing) = sum(A)
_t_weighted_sum(A::AbstractMatrix, w::AbstractVector) = sum(w' .* A)
_t_sample_size(Z, ::Nothing) = size(Z, 2)
_t_sample_size(Z, w::AbstractVector) = sum(w)
function _t_copula_loglik_factor(ν, L, Z; weights=nothing)
    d = size(Z, 1)
    n = _t_sample_size(Z, weights)
    Ltri = LinearAlgebra.LowerTriangular(L)
    Y = Ltri \ Z
    q = vec(sum(abs2, Y; dims=1))
    logdetR = 2 * sum(log, LinearAlgebra.diag(L))
    logconstant = SpecialFunctions.loggamma((ν + d) / 2) + (d - 1) * SpecialFunctions.loggamma(ν / 2) - d * SpecialFunctions.loggamma((ν + 1) / 2)
    joint = n * logconstant - (n / 2) * logdetR - (ν + d) / 2 * _t_weighted_sum(log1p.(q ./ ν), weights)
    marginals = (ν + 1) / 2 * _t_weighted_sum(log1p.(abs2.(Z) ./ ν), weights)
    return joint + marginals
end
function _fit_t_corr_given_nu(U, ν; weights=nothing)
    d = size(U, 1)
    n = _t_sample_size(U, weights)
    # For fixed ν, Student scores are constant throughout
    # the correlation optimization. Paramorph owns the correlation chart;
    # this profile keeps only the Student-specific likelihood objective.
    Z = Distributions.quantile.(Distributions.TDist(ν), U)
    pΣ = Paramorph.Correlation(:Σ, d)
    R₀ = _score_corr_start(Z)
    α₀ = Paramorph.unconstrain(pΣ, (R₀,))
    objective = α -> begin
        L = Paramorph.correlation_factor(pΣ, α)
        diagL = LinearAlgebra.diag(L)
        all(x -> isfinite(x) && x > zero(x), diagL) ||
            return convert(eltype(α), Inf)
        Ltri = LinearAlgebra.LowerTriangular(L)
        Y = Ltri \ Z
        q = vec(sum(abs2, Y; dims=1))
        logdetR = 2 * sum(log, diagL)
        return (n/2) * logdetR + (ν + d) / 2 * _t_weighted_sum(log1p.(q ./ ν), weights)
    end
    res = Optim.optimize(
        objective,
        α₀,
        Optim.LBFGS();
        autodiff=ADTypes.AutoForwardDiff(),
    )
    α̂ = Optim.minimizer(res)
    L̂ = Paramorph.correlation_factor(pΣ, α̂)
    R̂ = L̂ * L̂'
    R̂ = (R̂ + R̂') / 2
    ll = _t_copula_loglik_factor(ν, L̂, Z; weights)
    return (ν=ν, Σ=R̂, loglikelihood=ll, result=res,)
end
function _t_profile_upper(loss; upper0 = 0.5, max_expand = 12,)
    upper = upper0
    fmid = loss(upper / 2)
    fupper = loss(upper)
    expansions = 0
    while isfinite(fupper) && (!isfinite(fmid) || fupper < fmid)
        expansions += 1
        expansions > max_expand && error("Could not bracket the Student profile likelihood",)
        upper *= 2
        fmid = fupper
        fupper = loss(upper)
    end
    return upper, expansions
end
function _fit(::Type{<:TCopula}, U, ::Val{:mle}; weights=nothing)
    # λ = 1 / ν.  The endpoint λ = 0 is the Gaussian limit ν = Inf.
    G = _fit(GaussianCopula, U, Val(:mle); weights)
    Σ_gaussian = only(Distributions.params(G))
    ll_gaussian = _weighted_loglikelihood(G, U, weights)
    profile_loss = λ -> begin
        iszero(λ) && return -ll_gaussian
        ν = inv(λ)
        fitν = _fit_t_corr_given_nu(U, ν; weights)
        return -fitν.loglikelihood
    end
    upper, _ = _t_profile_upper(profile_loss)
    resλ = Optim.optimize(profile_loss, zero(upper), upper, Optim.Brent(),)
    λ̂ = Optim.minimizer(resλ)
    ν̂_finite = inv(λ̂)
    finite = _fit_t_corr_given_nu(U, ν̂_finite; weights)
    ll_finite = finite.loglikelihood
    Tll = typeof(float(ll_gaussian))
    ll_tol = 100 * eps(Tll) * max(one(Tll), abs(ll_gaussian))
    use_gaussian_limit = ll_gaussian >= ll_finite - ll_tol
    if use_gaussian_limit
        ν̂ = Inf
        Σ̂ = copy(Σ_gaussian)
    else
        ν̂ = ν̂_finite
        Σ̂ = finite.Σ
    end
    return TCopula(ν̂, Σ̂)
end
function _fit(::Type{<:TCopula}, U, ::Val{:itau_irho}; weights=nothing)
    size(U, 1) == 2 || throw(ArgumentError("Student rank matching is only defined in dimension 2"))
    τ̂ = _rank_measure(Val(:itau), U, weights)[1, 2]
    ρ̂ = _rank_measure(Val(:irho), U, weights)[1, 2]
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
            fhi < 0 ? Inf : Roots.find_zero(objective, (middle, upper), Roots.Bisection())
        end
    end
    ν = isinf(logν) ? Inf : exp(logν)
    return TCopula{2}(ν, [1.0 r; r 1.0])
end

function _fit(::Type{<:TCopula}, U, m::Val{:itau}; weights=nothing)
    R = _nearest_correlation(sinpi.(_rank_measure(m, U, weights) ./ 2))
    L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(R)).L
    ll_gaussian = _weighted_loglikelihood(GaussianCopula(R), U, weights)
    profile_loss = λ -> begin
        if iszero(λ) return -ll_gaussian end
        ν = inv(λ)
        Z = Distributions.quantile.(Distributions.TDist(ν), U)
        ll = _t_copula_loglik_factor(ν, L, Z; weights)
        return isfinite(ll) ? -ll : Inf
    end
    upper, _ = _t_profile_upper(profile_loss)
    resλ = Optim.optimize(profile_loss, zero(upper), upper, Optim.Brent(),)
    λ̂ = Optim.minimizer(resλ)
    ll_finite = -profile_loss(λ̂)
    Tll = typeof(float(ll_gaussian))
    ll_tol = 100 * eps(Tll) * max(one(Tll), abs(ll_gaussian))
    ν̂ = ll_gaussian >= ll_finite - ll_tol ? Inf : inv(λ̂)
    return TCopula(ν̂, R)
end

_available_fitting_methods(::Type{<:TCopula}, d) = d == 2 ? (:mle, :itau, :itau_irho) : (:mle, :itau)
