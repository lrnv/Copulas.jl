###############################################################################
#####  Nataf correction.
#####  User-facing function: `Nataf(margins, R)`
#####
#####  Calibrates the correlation matrix of a Gaussian copula so that the
#####  SklarDist built from it and the given margins attains a target Pearson
#####  correlation matrix.
#####
#####  `_nataf_problem(Fᵢ, Fⱼ, ρ, nodes)` dispatches on the margin types to
#####  provide the attainable range and the appropriate inverse correlation map;
#####  `Nataf` performs the common validation and clamping around that map.
###############################################################################

# Generic problem: quadrature + bisection.
function _nataf_problem(Fᵢ::Distributions.UnivariateDistribution, Fⱼ::Distributions.UnivariateDistribution,
                        ρ::Real, nodes::Integer)
    T = typeof(ρ)
    # Probabilists' Gauss-Hermite rule from the Golub-Welsch eigenproblem.
    # The Float64 nodes are converted to the working type; quadrature error
    # dominates their initial rounding error.
    E = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(nodes), sqrt.(1.0:(nodes-1))))
    z, w = T.(E.values), T.(abs2.(E.vectors[1, :]))

    # Pull a margin back to normal space and standardize it using moments from
    # this same rule, so comonotone margins correlate to exactly one on the rule.
    function standardized(F)
        μ, σ = Distributions.mean(F), Distributions.std(F)
        (isfinite(μ) && isfinite(σ) && σ > 0) || throw(ArgumentError(
            "The Nataf correction is only defined for margins with a finite mean and a finite positive " *
            "standard deviation, but $(F) has mean $(μ) and standard deviation $(σ)."))
        q(t) = Distributions.quantile(F,
            clamp(StatsFuns.normcdf(t), nextfloat(zero(T)), prevfloat(one(T))))
        μ̂ = sum(w[a] * q(z[a]) for a in eachindex(z))
        σ̂ = sqrt(sum(w[a] * abs2(q(z[a]) - μ̂) for a in eachindex(z)))
        return t -> (q(t) - μ̂) / σ̂
    end
    gᵢ, gⱼ = standardized(Fᵢ), standardized(Fⱼ)
    gᵢ_at_z = gᵢ.(z)

    # The conditional form zⱼ = ρ₀zₐ + √(1-ρ₀²)z_b turns the correlated
    # bivariate expectation into a product rule over independent normals.
    function induced(ρ₀)
        s, r = sqrt(max(zero(T), one(T) - ρ₀^2)), zero(T)
        @inbounds for a in eachindex(z)
            inner = zero(T)
            for b in eachindex(z)
                inner += w[b] * gⱼ(ρ₀ * z[a] + s * z[b])
            end
            r += w[a] * gᵢ_at_z[a] * inner
        end
        return r
    end

    lo, hi = induced(-one(T)), induced(one(T))
    inverse(target) = Roots.find_zero(
        ρ₀ -> induced(ρ₀) - target,
        (nextfloat(-one(T)), prevfloat(one(T))), Roots.Bisection())
    return (; lo, hi, inverse)
end

# Closed-form problems. Their inverse maps are exact in the working type, so
# BigFloat inputs give full-precision results on these paths.
function _nataf_problem(::Distributions.Normal, ::Distributions.Normal,
                        ρ::Real, nodes::Integer)
    # Pearson correlation is invariant under affine margins, so the target is the parameter.
    return (; lo=-one(ρ), hi=one(ρ), inverse=identity)
end
function _nataf_problem(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.LogNormal,
                        ρ::Real, nodes::Integer)
    # r(ρ₀) = (exp(ρ₀sᵢsⱼ) - 1) / √((exp(sᵢ²) - 1)(exp(sⱼ²) - 1)), independent of the μ's.
    sᵢ, sⱼ = oftype(ρ, Distributions.params(Fᵢ)[2]), oftype(ρ, Distributions.params(Fⱼ)[2])
    D = sqrt(expm1(sᵢ^2) * expm1(sⱼ^2))
    lo, hi = expm1(-sᵢ * sⱼ) / D, expm1(sᵢ * sⱼ) / D
    inverse(ρ) = log1p(ρ * D) / (sᵢ * sⱼ)
    return (; lo, hi, inverse)
end
function _nataf_problem(::Distributions.Normal, Fⱼ::Distributions.LogNormal,
                        ρ::Real, nodes::Integer)
    # The Normal margin is affine in its germ, so r(ρ₀) = ρ₀ s/√(exp(s²) - 1) is linear.
    s = oftype(ρ, Distributions.params(Fⱼ)[2])
    b = s / sqrt(expm1(s^2))
    return (; lo=-b, hi=b, inverse=ρ -> ρ / b)
end
function _nataf_problem(::Distributions.Uniform, ::Distributions.Uniform,
                        ρ::Real, nodes::Integer)
    # Pearson correlation of Gaussian-copula uniforms is Spearman's rho:
    # r(ρ₀) = 6 asin(ρ₀/2) / π.
    return (; lo=-one(ρ), hi=one(ρ), inverse=r -> 2sinpi(r / 6))
end
function _nataf_problem(::Distributions.Uniform, ::Distributions.Normal,
                        ρ::Real, nodes::Integer)
    # r(ρ₀) = ρ₀ √(3/π).
    b = sqrt(oftype(ρ, 3) / oftype(ρ, π))
    return (; lo=-b, hi=b, inverse=r -> r / b)
end
function _nataf_problem(::Distributions.Uniform, Fⱼ::Distributions.LogNormal,
                        ρ::Real, nodes::Integer)
    # r(ρ₀) = 2√3 (Φ(sρ₀/√2) - 1/2) / √(exp(s²) - 1).
    s = oftype(ρ, Distributions.params(Fⱼ)[2])
    half, root2 = one(ρ) / 2, sqrt(oftype(ρ, 2))
    scale = 2sqrt(oftype(ρ, 3)) / sqrt(expm1(s^2))
    induced(ρ₀) = scale * (StatsFuns.normcdf(s * ρ₀ / root2) - half)
    inverse(r) = root2 / s * StatsFuns.norminvcdf(half + r / scale)
    return (; lo=induced(-one(ρ)), hi=induced(one(ρ)), inverse)
end
# The induced correlation map is symmetric in the pair, so reversed-order
# methods forward to the corresponding implementation above.
function _nataf_problem(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.Normal,
                        ρ::Real, nodes::Integer)
    return _nataf_problem(Fⱼ, Fᵢ, ρ, nodes)
end
function _nataf_problem(Fᵢ::Distributions.Normal, Fⱼ::Distributions.Uniform,
                        ρ::Real, nodes::Integer)
    return _nataf_problem(Fⱼ, Fᵢ, ρ, nodes)
end
function _nataf_problem(Fᵢ::Distributions.LogNormal, Fⱼ::Distributions.Uniform,
                        ρ::Real, nodes::Integer)
    return _nataf_problem(Fⱼ, Fᵢ, ρ, nodes)
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
(``\\rho_0 = \\log(1 + \\rho\\sqrt{(e^{s_i^2}-1)(e^{s_j^2}-1)})/(s_is_j)``), and all
pairs among `Normal`, `LogNormal`, and `Uniform` margins. Because non-Gaussian
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
    # Absorb floating-point noise in attainable bounds and snap boundary targets
    # to ±1 instead of evaluating an inverse at a rounded endpoint.
    T = float(mapreduce(Distributions.partype, promote_type, margins; init=eltype(R)))
    tol = eps(T)^(2//3)
    R₀ = Matrix{T}(LinearAlgebra.I, d, d)
    for i in 1:d, j in (i+1):d
        ρ = T(R[i, j])
        if iszero(ρ)
            ρ₀ = zero(T)
        else
            problem = _nataf_problem(margins[i], margins[j], ρ, nodes)
            problem.lo - tol <= ρ <= problem.hi + tol || throw(ArgumentError(
                "The target Pearson correlation $(ρ) for margins ($(i), $(j)) is outside the range " *
                "[$(round(problem.lo, digits=4)), $(round(problem.hi, digits=4))] that these margins can attain. " *
                "Pearson correlations of non-Gaussian margins cannot reach all of [-1, 1] " *
                "(Fréchet-Hoeffding bounds), so the target itself has to change."))
            ρ₀ = ρ >= problem.hi - tol ? one(T) :
                 ρ <= problem.lo + tol ? -one(T) :
                 clamp(problem.inverse(ρ), -one(T), one(T))
        end
        R₀[i, j] = R₀[j, i] = ρ₀
    end
    return R₀
end
function Nataf(margins, ρ::Real; nodes::Integer=32)
    ρf = float(ρ)
    return Nataf(margins, [one(ρf) ρf; ρf one(ρf)]; nodes=nodes)[1, 2]
end
