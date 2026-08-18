###############################################################################
#####  Nataf correction.
#####  User-facing function: `Nataf(margins, R)`
#####
#####  Calibrates the correlation matrix of a Gaussian copula so that the
#####  SklarDist built from it and the given margins attains a target Pearson
#####  correlation matrix.
#####
#####  The pairwise work is done by `_nataf_pair(Fᵢ, Fⱼ, ρ, i, j, nodes)`,
#####  dispatching on the margin types: the generic method inverts the induced
#####  correlation map numerically, and pairs with an analytic map (Normal and
#####  LogNormal combinations) have specialized closed-form methods.
###############################################################################

# n-point probabilists' Gauss-Hermite rule from the Golub-Welsch eigenvalue
# problem. Weights sum to one, so sum(w .* f.(z)) approximates E[f(Z)], Z ~ N(0,1).
# The eigensolver works in Float64; callers convert the nodes to their working
# type, which is fine since the quadrature truncation error dominates anyway.
function _gauss_hermite(n::Integer)
    n >= 2 || throw(ArgumentError("The Nataf correction needs at least 2 quadrature nodes, got nodes=$(n)."))
    E = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(n), sqrt.(1.0:(n-1))))
    return E.values, abs2.(E.vectors[1, :])
end

# The margin pulled back to standard normal space, z ↦ F⁻¹(Φ(z)), standardized
# by its moments *under the quadrature rule* so that comonotone margins map to
# a correlation of exactly one on the rule itself.
function _nataf_standardized(F::Distributions.UnivariateDistribution, k::Integer, z, w)
    T = eltype(z)
    μ, σ = Distributions.mean(F), Distributions.std(F)
    (isfinite(μ) && isfinite(σ) && σ > 0) || throw(ArgumentError(
        "The Nataf correction is only defined for margins with a finite mean and a finite positive " *
        "standard deviation, but margin $(k) ($(F)) has mean $(μ) and standard deviation $(σ)."))
    # The clamp keeps extreme quadrature nodes away from the exact 0 and 1 that
    # would send a quantile to ±∞.
    q(t) = Distributions.quantile(F, clamp(StatsFuns.normcdf(t), nextfloat(zero(T)), prevfloat(one(T))))
    μ̂ = sum(w[a] * q(z[a]) for a in eachindex(z))
    σ̂ = sqrt(sum(w[a] * abs2(q(z[a]) - μ̂) for a in eachindex(z)))
    return t -> (q(t) - μ̂) / σ̂
end

# Pearson correlation induced between margins i and j by germ correlation ρ₀:
# the conditional form zⱼ = ρ₀zₐ + √(1-ρ₀²)z_b turns the correlated bivariate
# expectation into a product rule over two independent standard normals.
function _nataf_induced(gᵢ_at_z, gⱼ, z, w, ρ₀::T) where {T<:Real}
    s = sqrt(max(zero(T), one(T) - ρ₀^2))
    r = zero(T)
    @inbounds for a in eachindex(z)
        inner = zero(T)
        for b in eachindex(z)
            inner += w[b] * gⱼ(ρ₀ * z[a] + s * z[b])
        end
        r += w[a] * gᵢ_at_z[a] * inner
    end
    return r
end

# The attainable Pearson range is [ρ(-1), ρ(1)] (Fréchet-Hoeffding bounds of the
# margins). The tolerance adapts to the working precision (≈ 4e-11 in Float64)
# and absorbs floating-point noise in the bounds, so boundary targets snap to
# ρ₀ = ±1 instead of bisecting or throwing.
_nataf_tol(::Type{T}) where {T<:Real} = eps(T)^(2//3)
function _nataf_checkrange(ρ::T, lo::T, hi::T, i::Integer, j::Integer) where {T<:Real}
    lo - _nataf_tol(T) <= ρ <= hi + _nataf_tol(T) || throw(ArgumentError(
        "The target Pearson correlation $(ρ) for margins ($(i), $(j)) is outside the range " *
        "[$(round(lo, digits=4)), $(round(hi, digits=4))] that these margins can attain. " *
        "Pearson correlations of non-Gaussian margins cannot reach all of [-1, 1] " *
        "(Fréchet-Hoeffding bounds), so the target itself has to change."))
end

_nataf_promote(Fᵢ, Fⱼ, ρ) = float(promote_type(typeof(ρ), Distributions.partype(Fᵢ), Distributions.partype(Fⱼ)))

# Generic fallback: quadrature + bisection.
function _nataf_pair(Fᵢ::Distributions.UnivariateDistribution, Fⱼ::Distributions.UnivariateDistribution,
                     ρ::Real, i::Integer, j::Integer, nodes::Integer)
    T = _nataf_promote(Fᵢ, Fⱼ, ρ)
    ρ, tol = T(ρ), _nataf_tol(T)
    iszero(ρ) && return zero(T)
    z64, w64 = _gauss_hermite(nodes)
    z, w = T.(z64), T.(w64)
    gᵢ, gⱼ = _nataf_standardized(Fᵢ, i, z, w), _nataf_standardized(Fⱼ, j, z, w)
    gᵢ_at_z = gᵢ.(z)
    lo = _nataf_induced(gᵢ_at_z, gⱼ, z, w, -one(T))
    hi = _nataf_induced(gᵢ_at_z, gⱼ, z, w, one(T))
    _nataf_checkrange(ρ, lo, hi, i, j)
    ρ >= hi - tol && return one(T)
    ρ <= lo + tol && return -one(T)
    # The induced correlation is increasing in ρ₀, so bisection cannot lose the root.
    bracket = (nextfloat(-one(T)), prevfloat(one(T)))
    return Roots.find_zero(ρ₀ -> _nataf_induced(gᵢ_at_z, gⱼ, z, w, ρ₀) - ρ, bracket, Roots.Bisection())
end

# Closed forms. Each is exact in the working type, so BigFloat inputs give
# full-precision results on these paths.
function _nataf_pair(Fᵢ::Distributions.Normal, Fⱼ::Distributions.Normal,
                     ρ::Real, i::Integer, j::Integer, nodes::Integer)
    # Pearson correlation is invariant under affine margins, so the target is the parameter.
    T = _nataf_promote(Fᵢ, Fⱼ, ρ)
    _nataf_checkrange(T(ρ), -one(T), one(T), i, j)
    return clamp(T(ρ), -one(T), one(T))
end
function _nataf_pair(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.LogNormal,
                     ρ::Real, i::Integer, j::Integer, nodes::Integer)
    # r(ρ₀) = (exp(ρ₀sᵢsⱼ) - 1) / √((exp(sᵢ²) - 1)(exp(sⱼ²) - 1)), independent of the μ's.
    T = _nataf_promote(Fᵢ, Fⱼ, ρ)
    sᵢ, sⱼ = T(Distributions.params(Fᵢ)[2]), T(Distributions.params(Fⱼ)[2])
    D = sqrt(expm1(sᵢ^2) * expm1(sⱼ^2))
    _nataf_checkrange(T(ρ), expm1(-sᵢ * sⱼ) / D, expm1(sᵢ * sⱼ) / D, i, j)
    return clamp(log1p(T(ρ) * D) / (sᵢ * sⱼ), -one(T), one(T))
end
function _nataf_pair(Fᵢ::Distributions.Normal, Fⱼ::Distributions.LogNormal,
                     ρ::Real, i::Integer, j::Integer, nodes::Integer)
    # The Normal margin is affine in its germ, so r(ρ₀) = ρ₀ s/√(exp(s²) - 1) is linear.
    T = _nataf_promote(Fᵢ, Fⱼ, ρ)
    s = T(Distributions.params(Fⱼ)[2])
    b = s / sqrt(expm1(s^2))
    _nataf_checkrange(T(ρ), -b, b, i, j)
    return clamp(T(ρ) / b, -one(T), one(T))
end
# The induced correlation map is symmetric in the pair, so the reversed order
# forwards to the method above (the i, j indices keep the caller's order for
# error messages).
function _nataf_pair(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.Normal,
                     ρ::Real, i::Integer, j::Integer, nodes::Integer)
    return _nataf_pair(Fⱼ, Fᵢ, ρ, i, j, nodes)
end

"""
    Nataf(margins, R; nodes=32)
    Nataf(margins, ρ::Real; nodes=32)

Nataf correction [nataf1962, liu1986](@cite): compute the correlation matrix for a
[`GaussianCopula`](@ref) such that the [`SklarDist`](@ref) built from it and the given
`margins` has Pearson correlation matrix `R`.

A Gaussian copula with parameter ``\\rho_0`` induces, once the margins are applied, a
Pearson correlation that depends on the shape of the margins and equals ``\\rho_0`` only
when the margins are themselves Gaussian. Matching a Pearson target ``\\rho`` therefore
means inverting, for each pair of margins,

```math
\\rho(\\rho_0) = \\mathbb E\\left[g_i(Z_i)\\,g_j(Z_j)\\right],
\\quad (Z_i, Z_j) \\sim \\mathcal N\\left(0, \\begin{pmatrix} 1 & \\rho_0 \\\\ \\rho_0 & 1\\end{pmatrix}\\right),
```

where ``g_k(z) = (F_k^{-1}(\\Phi(z)) - \\mu_k)/\\sigma_k`` is the standardized margin pulled
back to standard normal space. The expectation is evaluated with a product Gauss-Hermite
rule and, since it is increasing in ``\\rho_0``, inverted by bisection.

# Arguments
- `margins`: a `Tuple` or vector of univariate distributions, each with finite mean and
  finite positive standard deviation.
- `R`: the target Pearson correlation matrix (or a single target correlation `ρ` when
  there are exactly two margins, in which case the corrected scalar is returned).
- `nodes`: number of Gauss-Hermite nodes per dimension. The default is accurate to about
  `1e-8` for well-behaved margins; heavy-tailed or strongly skewed margins converge more
  slowly and want more nodes.

Zero targets map to exactly zero. Pairs whose induced correlation is known analytically
skip the quadrature and use the closed form instead: `Normal`-`Normal` pairs (the
identity, so Gaussian margins reproduce `R` exactly), `LogNormal`-`LogNormal` pairs
(``\\rho_0 = \\log(1 + \\rho\\sqrt{(e^{s_i^2}-1)(e^{s_j^2}-1)})/(s_is_j)``), and mixed
`Normal`-`LogNormal` pairs (``\\rho_0 = \\rho\\sqrt{e^{s^2}-1}/s``). Because non-Gaussian
margins cannot attain every Pearson correlation (the Fréchet-Hoeffding bounds), a target
outside the attainable range throws an error naming the pair and the range. The corrected
matrix is not guaranteed to stay positive definite for extreme targets; the
`GaussianCopula` constructor validates it.

The computation is type-generic and follows the precision of the inputs: `BigFloat`
targets or margin parameters yield `BigFloat` results, at full precision on the
closed-form paths (the quadrature nodes of the generic path are computed in `Float64`).

# Example

```julia
using Copulas, Distributions, Statistics

m  = (LogNormal(0, 0.8), Gamma(1, 2), Beta(1, 2))
R0 = [1 0.7 0.3; 0.7 1 0.5; 0.3 0.5 1]

D = SklarDist(GaussianCopula(Nataf(m, R0)), m)
cor(rand(D, 10^6)') # ≈ R0, while GaussianCopula(R0) directly would miss the target.
```

References:
* [nataf1962](@cite) Nataf, A. (1962). Détermination des distributions de probabilités dont les marges sont données.
* [liu1986](@cite) Liu, P.-L., & Der Kiureghian, A. (1986). Multivariate distribution models with prescribed marginals and covariances.
"""
function Nataf(margins, R::AbstractMatrix{<:Real}; nodes::Integer=32)
    d = length(margins)
    all(m -> m isa Distributions.UnivariateDistribution, margins) || throw(ArgumentError(
        "margins must be univariate distributions, got $(typeof(margins))."))
    size(R) == (d, d) || throw(ArgumentError(
        "Got $(d) margins for a correlation matrix of size $(size(R))."))
    LinearAlgebra.issymmetric(R) || throw(ArgumentError("The target correlation matrix must be symmetric."))
    all(isapprox.(LinearAlgebra.diag(R), 1)) || throw(ArgumentError("The target correlation matrix must have a unit diagonal."))
    nodes >= 2 || throw(ArgumentError("The Nataf correction needs at least 2 quadrature nodes, got nodes=$(nodes)."))
    T = float(mapreduce(Distributions.partype, promote_type, margins; init=eltype(R)))
    R₀ = Matrix{T}(LinearAlgebra.I, d, d)
    for i in 1:d, j in (i+1):d
        R₀[i, j] = R₀[j, i] = _nataf_pair(margins[i], margins[j], T(R[i, j]), i, j, nodes)
    end
    return R₀
end
function Nataf(margins, ρ::Real; nodes::Integer=32)
    length(margins) == 2 || throw(ArgumentError(
        "A scalar Pearson target needs exactly 2 margins, got $(length(margins)). Pass a full correlation matrix instead."))
    -1 <= ρ <= 1 || throw(ArgumentError("The target correlation must lie in [-1, 1], got $(ρ)."))
    ρf = float(ρ)
    return Nataf(margins, [one(ρf) ρf; ρf one(ρf)]; nodes=nodes)[1, 2]
end
