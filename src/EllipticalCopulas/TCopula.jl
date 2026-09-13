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

Degrees of freedom must be positive. Covariance-like matrix inputs are
normalized to correlation scale, and non-positive-definite matrices are
rejected. Unlike the Gaussian copula, finite degrees of freedom produce
symmetric lower- and upper-tail dependence. Large `ν` approaches the Gaussian
copula and can be weakly identified.

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



U(C::TCopula) = isinf(C.df) ? Distributions.Normal() : Distributions.TDist(C.df)
N(C::TCopula) = isinf(C.df) ? Distributions.MvNormal : (Σ -> Distributions.MvTDist(C.df, Σ))
@inline _gaussian_limit(C::TCopula) = GaussianCopula(copy(C.Σ))
function _cdf(C::TCopula{d}, u) where d
    isinf(C.df) && return _cdf(_gaussian_limit(C), u)
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
    isinf(C.df) && return rosenblatt(_gaussian_limit(C), u)
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
    isinf(C.df) && return inverse_rosenblatt(_gaussian_limit(C), s)
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
    isinf(C.df) && return distortion(_gaussian_limit(C), js, uⱼₛ, i,)
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
    if isinf(C.df)
        Gcond, distortions = _conditional_components(_gaussian_limit(C), js, uⱼₛ, is,)
        return TCopula(Inf, copy(Gcond.Σ),), distortions
    end
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
function _t_copula_loglik_factor(ν, L, Z,)
    d, n = size(Z)
    Ltri = LinearAlgebra.LowerTriangular(L)
    # R = L L', hence
    # qᵢ = zᵢ' R⁻¹ zᵢ = ||L⁻¹ zᵢ||².
    Y = Ltri \ Z
    q = vec(sum(abs2, Y; dims=1))
    logdetR = 2 * sum(log, LinearAlgebra.diag(L))
    logconstant = SpecialFunctions.loggamma((ν + d) / 2) + (d - 1) * SpecialFunctions.loggamma(ν / 2) - d * SpecialFunctions.loggamma((ν + 1) / 2)
    joint = n * logconstant - (n / 2) * logdetR - (ν + d) / 2 * sum(log1p.(q ./ ν))
    marginals = (ν + 1) / 2 * sum(log1p.(abs2.(Z) ./ ν))
    return joint + marginals
end
function _fit_t_corr_given_nu(U, ν,)
    d, n = size(U)
    # For fixed ν, Student scores are constant throughout
    # the correlation optimization.
    Z = Distributions.quantile.(Distributions.TDist(ν), U)
    R₀ = _score_corr_start(Z)
    α₀ = _unbound_corr_params(d, R₀)
    objective = α -> begin
        L = _rebound_corr_factor(d, α)
        Ltri = LinearAlgebra.LowerTriangular(L)
        Y = Ltri \ Z
        q = vec(sum(abs2, Y; dims=1))
        logdetR = 2 * sum(log, LinearAlgebra.diag(L))
        return (n/2) * logdetR + (ν + d) / 2 * sum(log1p.(q ./ ν))
    end
    res = try
        Optim.optimize(objective, α₀, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff(),)
    catch
        Optim.optimize(objective, α₀, Optim.NelderMead(),)
    end
    α̂ = Optim.minimizer(res)
    L̂ = _rebound_corr_factor(d, α̂)
    R̂ = L̂ * L̂'
    R̂ = (R̂ + R̂') / 2
    ll = _t_copula_loglik_factor(ν, L̂, Z,)
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
function _fit(::Type{<:TCopula}, U, ::Val{:mle},)
    # λ = 1 / ν.  The endpoint λ = 0 is the Gaussian limit ν = Inf.
    G, gaussian_details = _fit(GaussianCopula, U, Val(:mle))
    Σ_gaussian = Distributions.params(G).Σ
    ll_gaussian = Distributions.loglikelihood(G, U)
    profile_evaluations = Ref(0)
    profile_loss = λ -> begin
        profile_evaluations[] += 1
        if iszero(λ) return -ll_gaussian end
        ν = inv(λ)
        try
            fitν = _fit_t_corr_given_nu(U, ν)
            return -fitν.loglikelihood
        catch
            return Inf
        end
    end
    upper, expansions = _t_profile_upper(profile_loss)
    resλ = Optim.optimize(profile_loss, zero(upper), upper, Optim.Brent(),)
    λ̂ = Optim.minimizer(resλ)
    ν̂_finite = inv(λ̂)
    finite = _fit_t_corr_given_nu(U, ν̂_finite,)
    ll_finite = finite.loglikelihood
    # Numerical tolerance only for deciding whether the profile maximum
    # is distinguishable from the exact Gaussian endpoint.
    Tll = typeof(float(ll_gaussian))
    ll_tol = 100 * eps(Tll) * max(one(Tll), abs(ll_gaussian))
    use_gaussian_limit = ll_gaussian >= ll_finite - ll_tol
    if use_gaussian_limit
        ν̂ = Inf
        Σ̂ = copy(Σ_gaussian)
        converged = Optim.converged(resλ) && gaussian_details.converged
        correlation_iterations = gaussian_details.iterations
        correlation_optimizer = gaussian_details.optimizer
    else
        ν̂ = ν̂_finite
        Σ̂ = finite.Σ
        converged = Optim.converged(resλ) && Optim.converged(finite.result)
        correlation_iterations = Optim.iterations(finite.result)
        correlation_optimizer = Optim.summary(finite.result)
    end
    C = TCopula(ν̂, Σ̂)
    θ̂ = (; ν = ν̂, Σ = Σ̂)
    return C, (;θ̂, optimizer = "Brent(profile λ=1/ν)", correlation_optimizer, converged, iterations = Optim.iterations(resλ),
    profile_evaluations = profile_evaluations[], correlation_iterations, profile_upper = upper,
    profile_expansions = expansions, gaussian_limit = use_gaussian_limit,)
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
