"""
    EllipticalCopula

Internal abstract representation shared by elliptical copula implementations.
Users should construct and operate on documented concrete families such as
`GaussianCopula` and `TCopula`. Its type parameters and implementation hooks
are not public API; the current contributor architecture is described in the
developer guide.

See also: [`GaussianCopula`](@ref), [`TCopula`](@ref), [`U`](@ref), [`N`](@ref).
"""
abstract type EllipticalCopula{d,MT} <: Copula{d} end

"""
    U(C::EllipticalCopula)

Return the standardized univariate radial-family margin used to map latent
elliptical coordinates to uniforms. This internal family hook is consumed by
generic sampling and density code. Object-based methods may preserve runtime
parameters such as Student degrees of freedom; it is not public API.

See also: [`N`](@ref), [`EllipticalCopula`](@ref), [`GaussianCopula`](@ref),
[`TCopula`](@ref).
"""
U(C::CT) where {CT<:EllipticalCopula} = U(CT)

"""
    N(C::EllipticalCopula)

Return a constructor that maps a correlation matrix to the latent multivariate
elliptical distribution associated with `C`. Generic sampling and density code
requires consistency between this law and `U(C)`. Object-based methods may
capture runtime parameters. This is an internal, non-stable family hook.

See also: [`U`](@ref), [`EllipticalCopula`](@ref), [`GaussianCopula`](@ref),
[`TCopula`](@ref).
"""
N(C::CT) where {CT<:EllipticalCopula} = N(CT)

Base.eltype(C::EllipticalCopula) = Base.eltype(N(C)(C.Σ))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::AbstractMatrix{T}) where {T<:Real, CT<:EllipticalCopula}
    # More efficient version that precomputes stuff:
    n = N(C)(C.Σ)
    u = U(C)
    Random.rand!(rng,n,A)
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        A[i, j] = clamp(T(Distributions.cdf(u, A[i, j])), zero(T), one(T))
    end
    return A
end
function Distributions._logpdf(C::CT, u) where {CT <: EllipticalCopula}
    d = length(C)
    (u==zeros(d) || u==ones(d)) && return Inf 
    TΣ = eltype(C.Σ)
    Tu = eltype(u)
    T = promote_type(TΣ, Tu)
    U₁ = U(C)
    # quantiles
    x = Vector{T}(undef, d)
    @inbounds for i in 1:d
        x[i] = StatsBase.quantile(U₁, u[i])
    end
    # sum of univariate logpdfs
    s = zero(T)
    @inbounds for i in 1:d
        s += Distributions.logpdf(U₁, x[i])
    end
    return Distributions.logpdf(N(C)(C.Σ),x) - s
end
"""
    make_cor!(Σ)

Normalize the square covariance-like matrix `Σ` in place to unit diagonal by
the congruence transform `Σ[i,j] / sqrt(Σ[i,i]Σ[j,j])`. The diagonal entries
must be strictly positive and the element type must support the in-place
division. The function does not check symmetry, positive definiteness, or
whether the resulting entries form a valid correlation matrix; constructors
must perform the validation required by their family.

This is an internal constructor helper. Copy user-owned input before calling it
when mutation would be surprising.

See also: [`EllipticalCopula`](@ref), [`GaussianCopula`](@ref), [`TCopula`](@ref).
"""
function make_cor!(Σ)
    # Verify that Σ is a correlation matrix, otherwise make it so : 
    d = size(Σ,1)
    σ = [1/sqrt(Σ[i,i]) for i in 1:d]
    for i in 1:d
        for j in 1:d
            Σ[i,j] *= σ[i] .* σ[j]
        end
    end
end

# Multivariate Student probabilities through the normal scale-mixture
# representation. A fixed inner RNG makes the quadrature deterministic.
function _mvtcdf(df::Real, μ, Σ::AbstractMatrix, upper; rtol::Real=2e-6)
    q = length(upper)
    q == 0 && return 1.0

    dff = Float64(df)
    μf = Float64.(μ)
    upperf = Float64.(upper)
    Σf = Matrix{Float64}(LinearAlgebra.Symmetric(Matrix{Float64}(Σ)))

    if q == 1
        σ = sqrt(Σf[1, 1])
        return StatsFuns.tdistcdf(dff, (upperf[1] - μf[1]) / σ)
    end

    δ = upperf .- μf
    χ = Distributions.Chisq(dff)
    lo, hi = eps(Float64), 1.0 - eps(Float64)
    integrand(p) = begin
        w = Distributions.quantile(χ, clamp(Float64(p), lo, hi))
        b = sqrt(w / dff) .* δ
        MvNormalCDF.mvnormcdf(
            Σf, fill(-Inf, q), b; rng=Random.Xoshiro(0),
        )[1]
    end
    value = QuadGK.quadgk(integrand, 0.0, 1.0; rtol)[1]
    return clamp(value, 0.0, 1.0)
end

# ——————————————————————————————————————————————————————————
# Shared correlation-parameterization helpers (LKJ/partial corr)
# Map between correlation matrices and unconstrained vectors α ∈ ℝ^{d(d-1)/2}

# Gaussian / t (pareado por pares)

@inline function _unbound_corr_params(d::Int, Σ::AbstractMatrix)
    Lc = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ), check=true).L
    T = eltype(Σ)
    α = Vector{T}(undef, d*(d-1)÷2)
    k = 1
    @inbounds for i in 2:d
        denom = one(T)
        for j in 1:i-1
            z = Lc[i,j] / denom
            ϵ = sqrt(eps(T))
            z = clamp(z, -one(T) + ϵ, one(T) - ϵ)
            α[k] = atanh(z)
            k += 1
            denom *= sqrt(max(zero(T), one(T) - z*z))
        end
    end
    return α
end

@inline function _rebound_corr_factor(d::Int, α::AbstractVector{T}) where {T}
    L = zeros(T, d, d)
    L[1, 1] = one(T)
    k = 1
    @inbounds for i in 2:d
        denom = one(T)
        for j in 1:(i - 1)
            a = α[k]
            k += 1
            z = tanh(a)
            L[i, j] = z * denom
            denom *= inv(cosh(a)) # sqrt(1 - tanh(a)^2) = sech(a)
        end
        L[i, i] = denom
    end
    return L
end

function _score_corr_start(Z::AbstractMatrix)
    d = size(Z, 1)
    T = float(eltype(Z))
    R = Matrix{T}(Statistics.cor(Z; dims=2))

    if any(x -> !isfinite(x), R)
        return Matrix{T}(LinearAlgebra.I, d, d)
    end

    R = Matrix(LinearAlgebra.Symmetric((R + R') / 2))
    @inbounds for j in 1:d
        R[j, j] = one(T)
    end

    LinearAlgebra.isposdef(LinearAlgebra.Symmetric(R)) && return R

    I_d = Matrix{T}(LinearAlgebra.I, d, d)
    λ = sqrt(eps(T))

    while λ < one(T)
        Rλ = (one(T) - λ) .* R .+ λ .* I_d
        if LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Rλ))
            return Rλ
        end
        λ = min(one(T), 10λ)
    end

    return I_d
end

@inline function _rebound_corr_params(d::Int, α::AbstractVector{T}) where {T}
    L = _rebound_corr_factor(d, α)
    Σ = L * L'
    Σ = (Σ + Σ') / 2
    return Σ
end
