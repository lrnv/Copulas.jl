###############################################################################
#####  Nataf correction.
#####  User-facing function: `Nataf(margins, R)`
#####
#####  Calibrates the correlation matrix of a Gaussian copula so that the
#####  SklarDist built from it and the given margins attains a target Pearson
#####  correlation matrix.
###############################################################################

# n-point probabilists' Gauss-Hermite rule from the Golub-Welsch eigenvalue
# problem. Weights sum to one, so sum(w .* f.(z)) approximates E[f(Z)], Z ~ N(0,1).
function _gauss_hermite(n::Integer)
    n >= 2 || throw(ArgumentError("The Nataf correction needs at least 2 quadrature nodes, got nodes=$(n)."))
    E = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(n), sqrt.(1.0:(n-1))))
    return E.values, abs2.(E.vectors[1, :])
end

# The margin pulled back to standard normal space, z ↦ F⁻¹(Φ(z)), standardized
# by its moments *under the quadrature rule* so that comonotone margins map to
# a correlation of exactly one on the rule itself.
function _nataf_standardized(F::Distributions.UnivariateDistribution, k::Integer, z, w)
    μ, σ = Distributions.mean(F), Distributions.std(F)
    (isfinite(μ) && isfinite(σ) && σ > 0) || throw(ArgumentError(
        "The Nataf correction is only defined for margins with a finite mean and a finite positive " *
        "standard deviation, but margin $(k) ($(F)) has mean $(μ) and standard deviation $(σ)."))
    # The clamp keeps extreme quadrature nodes away from the exact 0 and 1 that
    # would send a quantile to ±∞.
    q(t) = Distributions.quantile(F, clamp(StatsFuns.normcdf(t), floatmin(Float64), 1 - eps()/2))
    μ̂ = sum(w[a] * q(z[a]) for a in eachindex(z))
    σ̂ = sqrt(sum(w[a] * abs2(q(z[a]) - μ̂) for a in eachindex(z)))
    return t -> (q(t) - μ̂) / σ̂
end

# Pearson correlation induced between margins i and j by germ correlation ρ₀:
# the conditional form zⱼ = ρ₀zₐ + √(1-ρ₀²)z_b turns the correlated bivariate
# expectation into a product rule over two independent standard normals.
function _nataf_induced(gᵢ_at_z, gⱼ, z, w, ρ₀::Float64)
    s = sqrt(max(0.0, 1 - ρ₀^2))
    r = 0.0
    @inbounds for a in eachindex(z)
        inner = 0.0
        for b in eachindex(z)
            inner += w[b] * gⱼ(ρ₀ * z[a] + s * z[b])
        end
        r += w[a] * gᵢ_at_z[a] * inner
    end
    return r
end

# The attainable Pearson range is [ρ(-1), ρ(1)] (Fréchet-Hoeffding bounds of the
# margins). The `tol` absorbs quadrature noise of a few ulps in bounds computed
# numerically, so boundary targets snap to ρ₀ = ±1 instead of bisecting or throwing.
const _NATAF_TOL = 1e-10
function _nataf_checkrange(ρ::Float64, lo::Float64, hi::Float64, i::Integer, j::Integer)
    lo - _NATAF_TOL <= ρ <= hi + _NATAF_TOL || throw(ArgumentError(
        "The target Pearson correlation $(ρ) for margins ($(i), $(j)) is outside the range " *
        "[$(round(lo, digits=4)), $(round(hi, digits=4))] that these margins can attain. " *
        "Pearson correlations of non-Gaussian margins cannot reach all of [-1, 1] " *
        "(Fréchet-Hoeffding bounds), so the target itself has to change."))
end

function _nataf_pair(ρ::Float64, gᵢ_at_z, gⱼ, z, w, i::Integer, j::Integer)
    iszero(ρ) && return 0.0
    lo = _nataf_induced(gᵢ_at_z, gⱼ, z, w, -1.0)
    hi = _nataf_induced(gᵢ_at_z, gⱼ, z, w, 1.0)
    _nataf_checkrange(ρ, lo, hi, i, j)
    ρ >= hi - _NATAF_TOL && return 1.0
    ρ <= lo + _NATAF_TOL && return -1.0
    # The induced correlation is increasing in ρ₀, so bisection cannot lose the root.
    return Roots.find_zero(ρ₀ -> _nataf_induced(gᵢ_at_z, gⱼ, z, w, ρ₀) - ρ, (-1.0, 1.0), Roots.Bisection())
end

# Closed-form corrections for pairs where the induced correlation is known
# analytically; they return `nothing` when no closed form applies, and the
# quadrature fallback takes over. The matrix method tries both argument orders,
# so each specialization only needs to be written once.
_nataf_exact(::Distributions.UnivariateDistribution, ::Distributions.UnivariateDistribution, ρ::Float64, i, j) = nothing
function _nataf_exact(::Distributions.Normal, ::Distributions.Normal, ρ::Float64, i, j)
    # Pearson correlation is invariant under affine margins, so the target is the parameter.
    _nataf_checkrange(ρ, -1.0, 1.0, i, j)
    return clamp(ρ, -1.0, 1.0)
end
function _nataf_exact(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.LogNormal, ρ::Float64, i, j)
    # r(ρ₀) = (exp(ρ₀sᵢsⱼ) - 1) / √((exp(sᵢ²) - 1)(exp(sⱼ²) - 1)), independent of the μ's.
    sᵢ, sⱼ = Distributions.params(Fᵢ)[2], Distributions.params(Fⱼ)[2]
    D = sqrt(expm1(sᵢ^2) * expm1(sⱼ^2))
    _nataf_checkrange(ρ, expm1(-sᵢ * sⱼ) / D, expm1(sᵢ * sⱼ) / D, i, j)
    return clamp(log1p(ρ * D) / (sᵢ * sⱼ), -1.0, 1.0)
end
function _nataf_exact(::Distributions.Normal, Fⱼ::Distributions.LogNormal, ρ::Float64, i, j)
    # The Normal margin is affine in its germ, so r(ρ₀) = ρ₀ s/√(exp(s²) - 1) is linear.
    s = Distributions.params(Fⱼ)[2]
    b = s / sqrt(expm1(s^2))
    _nataf_checkrange(ρ, -b, b, i, j)
    return clamp(ρ / b, -1.0, 1.0)
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
    # The quadrature is only built for margins involved in a pair without a
    # closed form, so e.g. all-Gaussian inputs never touch it.
    z, w = nothing, nothing
    g, g_at_z = Vector{Any}(nothing, d), Vector{Any}(nothing, d)
    function _quad!(k)
        z === nothing && ((z, w) = _gauss_hermite(nodes))
        if g[k] === nothing
            g[k] = _nataf_standardized(margins[k], k, z, w)
            g_at_z[k] = g[k].(z)
        end
    end
    R₀ = Matrix{Float64}(LinearAlgebra.I, d, d)
    for i in 1:d, j in (i+1):d
        ρ = Float64(R[i, j])
        r₀ = _nataf_exact(margins[i], margins[j], ρ, i, j)
        r₀ === nothing && (r₀ = _nataf_exact(margins[j], margins[i], ρ, i, j))
        if r₀ === nothing
            _quad!(i); _quad!(j)
            r₀ = _nataf_pair(ρ, g_at_z[i], g[j], z, w, i, j)
        end
        R₀[i, j] = R₀[j, i] = r₀
    end
    return R₀
end
function Nataf(margins, ρ::Real; nodes::Integer=32)
    length(margins) == 2 || throw(ArgumentError(
        "A scalar Pearson target needs exactly 2 margins, got $(length(margins)). Pass a full correlation matrix instead."))
    -1 <= ρ <= 1 || throw(ArgumentError("The target correlation must lie in [-1, 1], got $(ρ)."))
    return Nataf(margins, [1.0 Float64(ρ); Float64(ρ) 1.0]; nodes=nodes)[1, 2]
end
