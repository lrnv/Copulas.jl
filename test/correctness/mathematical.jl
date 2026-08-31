# Mathematical-path layer: expensive CDF/PDF, derivative, integral, rectangle,
# and transform equivalences are checked once per implementation mechanism,
# not for every parameterization of every public family.
# Classification inherited from the former generic suite:
# - universal invariants: copula margins, support and API identities live in
#   `contracts/copulas.jl`;
# - mechanism identities: derivatives, integrals, transforms and defining
#   representations are checked below on one representative implementation;
# - family formulas, limits and fixed regressions live in focused correctness,
#   equivalence, routing, or contract files according to what they prove.

# Smooth polynomial oracle. Its closed forms are independent of the generic
# integration, conditioning and Rosenblatt machinery exercised below.
struct PolynomialOracleCopula{d,T} <: Copulas.Copula{d}
    θ::T
end

@testset "Gaussian Sklar conditioning agrees with multivariate normal algebra" begin
    d = 3
    Σ = [1.0 0.7 0.3; 0.7 1.0 0.7; 0.3 0.7 1.0]
    μ = zeros(d)
    X = SklarDist(GaussianCopula{3}(Σ),
                  ntuple(i -> Normal(μ[i], Σ[i, i]), d))
    point = [0.2, 0.5, 0.8]
    expected, error = mvnormcdf(MvNormal(μ, Σ), fill(-Inf, d), point)
    @test cdf(X, point) ≈ expected atol=10sqrt(error)

    js, is, observed = 1:1, 2:3, [0.0]
    μcond = μ[is] + Σ[is, js] * (Σ[js, js] \ (observed - μ[js]))
    Σcond = Σ[is, is] - Σ[is, js] * (Σ[js, js] \ Σ[js, is])
    target = [-0.4, 0.7]
    expected_cond, cond_error = mvnormcdf(
        MvNormal(μcond, Σcond), fill(-Inf, 2), target)
    @test isapprox(cdf(condition(X, (1,), observed), target), expected_cond;
                   atol=max(10sqrt(cond_error), 5e-5), rtol=0)
end

# Same density, deliberately without a CDF method. It selects Copula.jl's
# generic density-integration route and therefore proves that route directly.
struct DensityOnlyPolynomialOracleCopula{d,T} <: Copulas.Copula{d}
    θ::T
end
Distributions.params(C::DensityOnlyPolynomialOracleCopula) = (; θ=C.θ)
Distributions._logpdf(C::DensityOnlyPolynomialOracleCopula, u) =
    log1p(C.θ * prod(1 .- 2 .* u))
PolynomialOracleCopula(θ) = PolynomialOracleCopula{2,typeof(θ)}(θ)
Distributions.params(C::PolynomialOracleCopula) = (; θ=C.θ)
function Copulas._cdf(C::PolynomialOracleCopula, u)
    return prod(u) * (1 + C.θ * prod(1 .- u))
end
function Distributions._logpdf(C::PolynomialOracleCopula, u)
    return log1p(C.θ * prod(1 .- 2 .* u))
end
_oracle_cdf(C, u) =
    prod(u) * (1 + C.θ * prod(1 .- u))
_oracle_pdf(C, u) =
    1 + C.θ * prod(1 .- 2 .* u)
_oracle_conditional_cdf(C::PolynomialOracleCopula, conditioned, target) =
    target * (1 + C.θ * (1 - 2conditioned) * (1 - target))

function Distributions._rand!(rng::Distributions.AbstractRNG,
                              C::PolynomialOracleCopula{d},
                              U::AbstractMatrix{T}) where {d,T<:Real}
    for j in axes(U, 2)
        for i in 1:(d - 1)
            U[i, j] = rand(rng)
        end
        conditioned = prod(1 - 2U[i, j] for i in 1:(d - 1))
        p = rand(rng)
        y = Roots.find_zero(
            target -> target * (1 + C.θ * conditioned * (1 - target)) - p,
            (zero(T), one(T)), Roots.Bisection())
        U[d, j] = y
    end
    return U
end

# Generator oracle: every derivative and inverse except ϕ itself must use the
# defaults from Generator.jl.
struct PowerExponentialOracleGenerator{T} <: Copulas.Generator
    θ::T
end
Copulas.ϕ(G::PowerExponentialOracleGenerator, t) = exp(-t^(inv(G.θ)))
Copulas.max_monotony(::PowerExponentialOracleGenerator) = Inf
Distributions.params(G::PowerExponentialOracleGenerator) = (; θ=G.θ)

# Tail oracle: A, mixed partials and the EV implementation must all be derived
# from this sole STDF definition.
struct LogisticOracleTail{T} <: Copulas.BivariatePickandsTail
    θ::T
end
Distributions.params(tail::LogisticOracleTail) = (; θ=tail.θ)
Copulas.ℓ(tail::LogisticOracleTail, x) =
    sum(xᵢ -> xᵢ^tail.θ, x)^(inv(tail.θ))
Copulas.A(tail::LogisticOracleTail, t::Real) =
    Copulas.ℓ(tail, (t, 1 - t))
Copulas._is_valid_in_dim(::LogisticOracleTail, d::Int) = d >= 2

# Complementary tail oracle: only Pickands' A is supplied, so ℓ and the first
# two Pickands derivatives must all use the generic BivariatePickandsTail API.
struct QuadraticPickandsOracleTail{T} <: Copulas.BivariatePickandsTail
    κ::T
end
Distributions.params(tail::QuadraticPickandsOracleTail) = (; κ=tail.κ)
Copulas.A(tail::QuadraticPickandsOracleTail, t::Real) =
    1 - tail.κ * t * (1 - t)

# Differentiate once in every coordinate without using the nested-copula
# Faà di Bruno implementation. This is intentionally small: family variants and
# censored/deep-tree regressions belong to the family and dispatch layers.
function _oracle_mixed_partial(f, u, coordinates=eachindex(u))
    function recurse(k, x)
        k > length(coordinates) && return f(x)
        i = coordinates[k]
        return ForwardDiff.derivative(x[i]) do value
            T = promote_type(typeof(value), eltype(x))
            next = T[j == i ? value : x[j] for j in eachindex(x)]
            recurse(k + 1, next)
        end
    end
    return recurse(1, u)
end

@testset "generic smooth-copula oracle" begin
    C = PolynomialOracleCopula(0.4)
    u = [0.37, 0.68]
    @test cdf(C, u) ≈ _oracle_cdf(C, u)
    @test pdf(C, u) ≈ _oracle_pdf(C, u)
    @test cdf(C, [u[1], 1.0]) ≈ u[1]
    @test cdf(C, [1.0, u[2]]) ≈ u[2]
    @test max(sum(u) - 1, 0) <= cdf(C, u) <= minimum(u)

    # Bypass the analytic _cdf method and exercise Copula.jl's density integral.
    integrated = invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
    @test integrated ≈ _oracle_cdf(C, u) atol=2e-5

    lower = [0.15, 0.25]
    upper = [0.55, 0.65]
    oracle_rectangle = (
        _oracle_cdf(C, upper) - _oracle_cdf(C, [lower[1], upper[2]]) -
        _oracle_cdf(C, [upper[1], lower[2]]) + _oracle_cdf(C, lower)
    )
    @test Copulas.measure(C, lower, upper) ≈ oracle_rectangle
    split = 0.4
    @test Copulas.measure(C, lower, upper) ≈
          Copulas.measure(C, lower, [split, upper[2]]) +
          Copulas.measure(C, [split, lower[2]], upper)

    D = condition(C, 1, u[1])
    @test D isa Copulas.DistortionFromCop
    @test cdf(D, u[2]) ≈ _oracle_conditional_cdf(C, u[1], u[2])
    @test pdf(D, u[2]) ≈
          ForwardDiff.derivative(v -> _oracle_conditional_cdf(C, u[1], v), u[2])

    R = rosenblatt(C, u)
    @test R ≈ [u[1], _oracle_conditional_cdf(C, u[1], u[2])]
    @test inverse_rosenblatt(C, R) ≈ u atol=2e-6
    @test Copulas.ρ(C) ≈ C.θ / 3 atol=2e-5
    @test Copulas.β(C) ≈ C.θ / 4
    @test Copulas.τ(C) ≈ 2 * C.θ / 9 atol=3e-2

    gini_integrand(v) = (
        1 + minimum(v) - maximum(v) + abs(sum(v) - 1)
    ) / 2
    gini_expectation, _ = HCubature.hcubature(
        v -> gini_integrand(v) * _oracle_pdf(C, v), zeros(2), ones(2))
    @test Copulas.γ(C) ≈ (gini_expectation - 0.5) / 0.25 atol=3e-2
    entropy, _ = HCubature.hcubature(zeros(2), ones(2)) do v
        density = _oracle_pdf(C, v)
        -density * log(density)
    end
    @test Copulas.ι(C) ≈ entropy atol=3e-2
    @test Copulas.λₗ(C) ≈ 0 atol=1e-8
    @test Copulas.λᵤ(C) ≈ 0 atol=1e-8
    for measure in SCALAR_DEPENDENCE_MEASURES
        prove_dependence_route!(measure, C)
    end

    conditional_mass, _ = QuadGK.quadgk(y -> pdf(D, y), 0.0, 1.0)
    @test conditional_mass ≈ 1
    @test pdf(C, u) ≈ pdf(D, u[2])

    C3 = PolynomialOracleCopula{3,Float64}(0.4)
    conditioned = 0.41
    target = [0.37, 0.68]
    H = condition(C3, (3,), (conditioned,))
    expected_conditional = prod(target) * (
        1 + C3.θ * prod(1 .- target) * (1 - 2conditioned))
    @test cdf(H, target) ≈ expected_conditional
    @test pdf(H, target) ≈
          1 + C3.θ * prod(1 .- 2 .* target) * (1 - 2conditioned)

    # Independent multivariate oracles close the dimension-dependent generic
    # dependence routes. They integrate the analytic polynomial CDF/density,
    # never the production implementations of the measures themselves.
    d3 = 3
    cube0, cube1 = zeros(d3), ones(d3)
    gini_integrand3(x) = (
        1 + minimum(x) - maximum(x) +
        max(abs(sum(x) - d3 / 2) - (d3 - 2) / 2, 0.0)
    ) / 2
    integrals, _ = HCubature.hcubature(cube0, cube1; rtol=2e-5) do x
        distribution = _oracle_cdf(C3, x)
        density = _oracle_pdf(C3, x)
        [distribution, distribution * density,
         gini_integrand3(x) * density, -density * log(density)]
    end
    cdf_integral, concordance, gini3, entropy3 = integrals
    rho3 = (2^d3 * (d3 + 1) * cdf_integral - d3 - 1) /
           (2^d3 - d3 - 1)
    @test Copulas.ρ(C3) ≈ rho3 atol=3e-4

    tau3 = 2^d3 / (2^(d3 - 1) - 1) * concordance -
           1 / (2^(d3 - 1) - 1)
    @test Copulas.τ(C3) ≈ tau3 atol=4e-2

    midpoint = fill(0.5, d3)
    c0 = _oracle_cdf(C3, midpoint)
    survival0 = 0.0
    for mask in Iterators.product(ntuple(_ -> (false, true), d3)...)
        point = [mask[i] ? midpoint[i] : 1.0 for i in 1:d3]
        survival0 += (-1)^count(identity, mask) * _oracle_cdf(C3, point)
    end
    beta3 = (2.0^(d3 - 1) * c0 + survival0 - 1) /
            (2^(d3 - 1) - 1)
    @test Copulas.β(C3) ≈ beta3 atol=1e-12

    a3 = 1 / (d3 + 1) + inv(factorial(d3 + 1))
    b3 = (2 + 4.0^(1 - d3)) / 3
    @test Copulas.γ(C3) ≈ (gini3 - a3) / (b3 - a3) atol=4e-2

    @test Copulas.ι(C3) ≈ entropy3 atol=4e-2
    @test Copulas.λₗ(C3) ≈ 0 atol=1e-8
    @test Copulas.λᵤ(C3) ≈ 0 atol=1e-8
    for measure in SCALAR_DEPENDENCE_MEASURES
        prove_dependence_route!(measure, C3)
    end

    for d in (2, 3)
        density_only = DensityOnlyPolynomialOracleCopula{d,Float64}(0.4)
        point = collect(range(0.37, 0.73; length=d))
        @test isapprox(cdf(density_only, point),
                       _oracle_cdf(density_only, point); atol=3e-5)
        prove_dispatch_route!(:cdf, density_only, :generic_density_integral)
    end
end

@testset "Sklar change-of-variables identities" begin
    C = PolynomialOracleCopula(0.4)
    margins = (Normal(0.3, 1.2), Gamma(2.3, 0.8))
    D = SklarDist(C, margins)
    x = [0.1, 1.4]
    u = [cdf(margins[i], x[i]) for i in eachindex(x)]

    @test cdf(D, x) ≈ _oracle_cdf(C, u)
    @test pdf(D, x) ≈
          _oracle_pdf(C, u) * prod(pdf(margins[i], x[i]) for i in eachindex(x))
end

@testset "Liouville radial-Dirichlet identity" begin
    α = (0.8, 1.4)
    α₀ = sum(α)
    radial = Beta(2.3, 1.7)
    C = LiouvilleCopula{2}(WilliamsonGenerator(radial, α₀), α)
    u = [0.75, 0.80]
    margins = ntuple(i -> Copulas.𝒲₋₁(C.G, α[i]), 2)
    x = ntuple(i -> quantile(margins[i], 1 - u[i]), 2)
    direction = Beta(α...)

    # Directly integrate the defining R * Dirichlet representation. The
    # production bivariate CDF uses expectation dispatch on the radial law.
    integrand(r) = begin
        r <= sum(x) && return 0.0
        lo = cdf(direction, x[1] / r)
        hi = cdf(direction, 1 - x[2] / r)
        pdf(radial, r) * max(0.0, hi - lo)
    end
    expected, _ = QuadGK.quadgk(integrand, sum(x), 1.0)
    @test cdf(C, u) ≈ expected atol=2e-7

    # The copula density must be the mixed derivative of that independently
    # integrated CDF, including both non-integer marginal transformations.
    # The CDF itself contains adaptive quadrature and numerical marginal
    # inversions; a moderately wide stencil avoids differentiating their noise.
    h = 1e-2
    mixed = (
        cdf(C, u .+ (h, h)) - cdf(C, u .+ (h, -h)) -
        cdf(C, u .+ (-h, h)) + cdf(C, u .- (h, h))
    ) / (4h^2)
    @test pdf(C, u) ≈ mixed atol=5e-4 rtol=5e-4
    prove_dispatch_route!(:cdf, C, :radial_dirichlet_identity)
    prove_dispatch_route!(:logpdf, C, :radial_dirichlet_identity)
end

@testset "nested Archimedean composition identity" begin
    root = Copulas.ClaytonGenerator(1.5)
    left = Copulas.GumbelGenerator(2.0)
    right = Copulas.FrankGenerator(3.0)
    C = NestedArchimedeanCopula(root;
        children=[GumbelCopula{2}(2.0), FrankCopula{2}(3.0)])
    u = [0.23, 0.47, 0.71, 0.59]

    child_value(G, x, I) = Copulas.ϕ(G, sum(Copulas.ϕ⁻¹(G, x[i]) for i in I))
    nested_cdf(x) = Copulas.ϕ(root,
        Copulas.ϕ⁻¹(root, child_value(left, x, 1:2)) +
        Copulas.ϕ⁻¹(root, child_value(right, x, 3:4)))

    @test cdf(C, u) ≈ nested_cdf(u)
    expected_density = _oracle_mixed_partial(nested_cdf, u)
    @test pdf(C, u) ≈ expected_density atol=2e-8 rtol=2e-8
    prove_dispatch_route!(:cdf, C, :nested_composition_identity)
    prove_dispatch_route!(:logpdf, C, :nested_composition_identity)
end

@testset "independent multivariate density identities" begin
    u = [0.31, 0.53, 0.74]

    # Archimedean change of variables: the d-th generator derivative is the
    # radial density term and every inverse-generator derivative contributes a
    # marginal Jacobian. This oracle does not call the copula density method.
    for C in (ClaytonCopula{3}(1.5), GumbelCopula{3}(1.5))
        G = C.G
        t = sum(Copulas.ϕ⁻¹(G, p) for p in u)
        expected = Copulas.ϕ⁽ᵏ⁾(G, 3, t) *
                   prod(Copulas.ϕ⁻¹⁽¹⁾(G, p) for p in u)
        @test pdf(C, u) ≈ expected rtol=2e-10
        prove_dispatch_route!(:logpdf, C, :archimedean_change_of_variables)
    end

    # Extreme-value densities are the full mixed derivative of their defining
    # CDF. The logistic oracle uses only ℓ, so it exercises the generic
    # multivariate EV density construction independently.
    ev = ExtremeValueCopula{3}(LogisticOracleTail(1.5))
    @test pdf(ev, u) ≈ _oracle_mixed_partial(v -> cdf(ev, v), u) rtol=2e-8
    prove_dispatch_route!(:logpdf, ev, :ev_cdf_mixed_derivative)

    logev = LogCopula{3}(1.5)
    @test pdf(logev, u) ≈ _oracle_mixed_partial(v -> cdf(logev, v), u) rtol=2e-8
    prove_dispatch_route!(:logpdf, logev, :ev_cdf_mixed_derivative)

    # Elliptical copula density is the multivariate density divided by all
    # standardized marginal densities. Cover both normal and Student kernels.
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    gaussian = GaussianCopula{3}(copy(Σ))
    znormal = quantile.(Normal(), u)
    gaussian_expected = pdf(MvNormal(zeros(3), Σ), znormal) /
                        prod(pdf.(Normal(), znormal))
    @test pdf(gaussian, u) ≈ gaussian_expected rtol=2e-12
    prove_dispatch_route!(:logpdf, gaussian, :elliptical_change_of_variables)

    ν = 5.0
    student = TCopula{3}(ν, copy(Σ))
    marginal = TDist(ν)
    zstudent = quantile.(marginal, u)
    student_expected = pdf(MvTDist(ν, Σ), zstudent) /
                       prod(pdf.(marginal, zstudent))
    @test pdf(student, u) ≈ student_expected rtol=2e-12
    prove_dispatch_route!(:logpdf, student, :elliptical_change_of_variables)

    # Liouville's radial--Dirichlet density, including non-integer marginal
    # Williamson orders and their Jacobians.
    α = (0.8, 1.1, 1.3)
    liouville = LiouvilleCopula{3}(Copulas.ClaytonGenerator(1.0), α)
    α₀ = sum(α)
    radial = Copulas.𝒲₋₁(liouville.G, α₀)
    margins = ntuple(i -> Copulas.𝒲₋₁(liouville.G, α[i]), 3)
    x = ntuple(i -> quantile(margins[i], 1 - u[i]), 3)
    radius = sum(x)
    expected_logdensity = SpecialFunctions.loggamma(α₀) -
        sum(SpecialFunctions.loggamma, α) + logpdf(radial, radius) +
        (1 - α₀) * log(radius) +
        sum((α[i] - 1) * log(x[i]) - logpdf(margins[i], x[i]) for i in 1:3)
    @test logpdf(liouville, u) ≈ expected_logdensity rtol=2e-10

    # Independently integrate the defining R*Dirichlet survival event using a
    # direct simplex density (the implementation uses beta stick-breaking).
    direction = Dirichlet(collect(α))
    expected_cdf, _ = HCubature.hcubature(zeros(2), ones(2); rtol=1e-7) do z
        a, b = z
        (iszero(a) || isone(a) || iszero(b) || isone(b)) && return 0.0
        simplex = [a, (1 - a) * b, (1 - a) * (1 - b)]
        threshold = maximum(x[i] / simplex[i] for i in 1:3)
        pdf(direction, simplex) * (1 - a) * ccdf(radial, threshold)
    end
    @test cdf(liouville, u) ≈ expected_cdf atol=4e-5 rtol=4e-5
    prove_dispatch_route!(:cdf, liouville, :radial_dirichlet_identity)
    prove_dispatch_route!(:logpdf, liouville, :radial_dirichlet_identity)

    # With only the full interaction coefficient nonzero, multivariate FGM is
    # exactly the polynomial oracle above. This covers the composed polynomial
    # density route without repeating its implementation.
    fgm = FGMCopula{3}([0.0, 0.0, 0.0, 0.4])
    polynomial = PolynomialOracleCopula{3,Float64}(0.4)
    @test cdf(fgm, u) ≈ _oracle_cdf(polynomial, u)
    @test pdf(fgm, u) ≈ _oracle_pdf(polynomial, u)
    prove_dispatch_route!(:cdf, fgm, :polynomial_identity)
    prove_dispatch_route!(:logpdf, fgm, :polynomial_identity)

    # Survival composition has unit absolute Jacobian; its density is the
    # wrapped copula density evaluated at the reflected coordinates.
    parent = ClaytonCopula{3}(1.5)
    survival = SurvivalCopula{3}(parent, (1, 3))
    reflected = [1 - u[1], u[2], 1 - u[3]]
    @test pdf(survival, u) ≈ pdf(parent, reflected)
    prove_dispatch_route!(:logpdf, survival, :survival_jacobian_identity)
end

@testset "generic generator oracle" begin
    G = PowerExponentialOracleGenerator(1.5)
    a = inv(G.θ)
    for t in (0.2, 0.7, 1.4)
        p = exp(-t^a)
        first_derivative = -a * t^(a - 1) * p
        second_derivative = p * (
            a^2 * t^(2a - 2) - a * (a - 1) * t^(a - 2))
        @test Copulas.ϕ(G, t) == p
        inverse = Copulas.ϕ⁻¹(G, p)
        @test inverse ≈ (-log(p))^G.θ
        @test inverse ≈ t
        @test Copulas.ϕ⁽¹⁾(G, t) ≈ first_derivative
        @test Copulas.ϕ⁽ᵏ⁾(G, 2, t) ≈ second_derivative
        @test Copulas.ϕ⁻¹⁽¹⁾(G, p) ≈
              -G.θ * (-log(p))^(G.θ - 1) / p
    end

    exponential = PowerExponentialOracleGenerator(1.0)
    t = 0.7
    @test Copulas.ϕ⁽ᵏ⁾⁻¹(exponential, 2, exp(-t); start_at=t) ≈ t
end

@testset "generic tail and extreme-value oracle" begin
    tail = LogisticOracleTail(1.5)
    x = [0.4, 0.7]
    expected_ℓ = sum(x .^ tail.θ)^(inv(tail.θ))
    @test Copulas.ℓ(tail, x) ≈ expected_ℓ
    @test Copulas.A(tail, Tuple(x ./ sum(x))) ≈ expected_ℓ / sum(x)
    S = sum(x .^ tail.θ)
    first_x = x[1]^(tail.θ - 1) * S^(inv(tail.θ) - 1)
    mixed_xy = (1 - tail.θ) * prod(x .^ (tail.θ - 1)) *
               S^(inv(tail.θ) - 2)
    @test Copulas.ellpartial(tail, x, (1,)) ≈ first_x
    @test Copulas.ellpartial(tail, x, (1, 2)) ≈ mixed_xy
    @test maximum(x) <= Copulas.ℓ(tail, x) <= sum(x)
    @test Copulas.ℓ(tail, 1.7 .* x) ≈ 1.7 * Copulas.ℓ(tail, x)

    y = reverse(x) .+ 0.2
    λ = 0.37
    @test Copulas.ℓ(tail, λ .* x .+ (1 - λ) .* y) <=
          λ * Copulas.ℓ(tail, x) + (1 - λ) * Copulas.ℓ(tail, y)

    x3 = [0.4, 0.7, 1.1]
    S3 = sum(x3 .^ tail.θ)
    for I in ((1,), (1, 3), (1, 2, 3))
        k = length(I)
        coefficient = k == 1 ? one(tail.θ) :
            prod(1 - j * tail.θ for j in 1:(k - 1))
        expected = coefficient * S3^(inv(tail.θ) - k) *
                   prod(x3[i]^(tail.θ - 1) for i in I)
        @test Copulas.ellpartial(tail, x3, I) ≈ expected
    end

    C = ExtremeValueCopula{2}(tail)
    u = [0.37, 0.68]
    expected_cdf = exp(-sum((-log.(u)) .^ tail.θ)^(inv(tail.θ)))
    @test cdf(C, u) ≈ expected_cdf
    h = 1e-5
    mixed_difference = (
        cdf(C, u .+ (h, h)) - cdf(C, u .+ (h, -h)) -
        cdf(C, u .+ (-h, h)) + cdf(C, u .- (h, h))
    ) / (4h^2)
    @test pdf(C, u) ≈ mixed_difference atol=1e-4
    @test cdf(C, u .^ 1.7) ≈ cdf(C, u)^1.7

    C3 = ExtremeValueCopula{3}(tail)
    u3 = [0.37, 0.55, 0.73]
    @test cdf(C3, u3 .^ 1.7) ≈ cdf(C3, u3)^1.7

    pickands = QuadraticPickandsOracleTail(0.5)
    weight = 0.37
    expected_A = 1 - pickands.κ * weight * (1 - weight)
    @test Copulas.A(pickands, weight) == expected_A
    @test Copulas.dA(pickands, weight) ≈ pickands.κ * (2 * weight - 1)
    @test Copulas.d²A(pickands, weight) ≈ 2 * pickands.κ
    @test Copulas.ℓ(pickands, x) ≈ sum(x) * Copulas.A(pickands, x[1] / sum(x))

    pickands_copula = ExtremeValueCopula{2}(pickands)
    pickands_cdf(v) = exp(-sum(-log.(v)) *
        Copulas.A(pickands, -log(v[1]) / sum(-log.(v))))
    expected_density = ForwardDiff.hessian(pickands_cdf, u)[1, 2]
    @test cdf(pickands_copula, u) ≈ pickands_cdf(u)
    @test pdf(pickands_copula, u) ≈ expected_density atol=2e-6
end

@testset "generic Williamson oracle" begin
    radial = Uniform(1.0, 2.0)
    G = WilliamsonGenerator(radial, 3.0)
    t = 0.4
    expected = 1 - 2t * log(2) + t^2 / 2
    @test Copulas.ϕ(G, t) ≈ expected
    @test Copulas.𝒲₋₁(G, 3.0) === radial

    # exp(-t) is the Williamson transform of Gamma(d, 1) at every order d.
    # The real-order case also exercises the exact beta-product reduction.
    exponential = PowerExponentialOracleGenerator(1.0)
    for order in (3, 2.4)
        inverse = Copulas.𝒲₋₁(exponential, order)
        reference = Gamma(order, 1.0)
        # One interior point per order exercises the CDF/PDF mechanisms; the
        # distribution contracts cover their domains separately.
        for x in (1.2,)
            @test cdf(inverse, x) ≈ cdf(reference, x) atol=2e-7
            @test pdf(inverse, x) ≈ pdf(reference, x) atol=2e-7
        end
        for p in (0.6,)
            @test quantile(inverse, p) ≈ quantile(reference, p) atol=2e-6
        end
    end
end

@testset "Archimedean radial and Kendall representations" begin
    C = ClaytonCopula{2}(1.5)
    G = C.G
    U = rand(StableRNG(121), C, 300)
    radial_from_copula = vec(sum(Copulas.ϕ⁻¹.(Ref(G), U); dims=1))
    radial_direct = rand(StableRNG(122), Copulas.𝒲₋₁(G, 2), 300)
    @test pvalue(ApproximateTwoSampleKSTest(radial_from_copula, radial_direct)) > 1e-3
    @test pvalue(ApproximateTwoSampleKSTest(cdf(C, U), Copulas.ϕ.(Ref(G), radial_direct))) > 1e-3
end

@testset "extreme-value representation coherence" begin
    for (tail, d) in TAIL_CASES
        u = collect(range(0.35, 0.75; length=d))
        C = ExtremeValueCopula{d}(tail)
        @test cdf(C, u) ≈ exp(-Copulas.ℓ(tail, -log.(u)))
        power = 1.7
        @test cdf(C, u .^ power) ≈ cdf(C, u)^power
        prove_dispatch_route!(:cdf, C, :stable_tail_representation)
    end
end

@testset "Archimax defining formula" begin
    C = ArchimaxCopula{2}(Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))
    u = [0.37, 0.68]
    x = Copulas.ϕ⁻¹(C.gen, u[1])
    y = Copulas.ϕ⁻¹(C.gen, u[2])
    expected = Copulas.ϕ(C.gen, (x + y) * Copulas.A(C.tail, y / (x + y)))
    @test cdf(C, u) ≈ expected
end

@testset "copula volumes are inclusion-exclusion measures" begin
    C = PolynomialOracleCopula{3,Float64}(0.3)
    lower = [0.12, 0.18, 0.24]
    upper = [0.68, 0.73, 0.81]
    expected = sum(Iterators.product((0:1 for _ in 1:3)...)) do corner
        point = [corner[i] == 1 ? upper[i] : lower[i] for i in 1:3]
        (-1)^(3 - sum(corner)) * cdf(C, point)
    end
    @test Copulas.measure(C, lower, upper) ≈ expected atol=2e-8

    split = 0.46
    left_upper = copy(upper)
    left_upper[1] = split
    right_lower = copy(lower)
    right_lower[1] = split
    @test Copulas.measure(C, lower, upper) ≈
          Copulas.measure(C, lower, left_upper) +
          Copulas.measure(C, right_lower, upper) atol=2e-8
    independence = IndependentCopula{3}()
    @test Copulas.measure(independence, lower, upper) ≈ prod(upper - lower)
    u = [0.32, 0.54, 0.76]
    @test cdf(independence, u) == prod(u)
    @test logpdf(independence, u) == 0
    prove_dispatch_route!(:cdf, independence, :independence_product_identity)
    prove_dispatch_route!(:logpdf, independence, :independence_product_identity)
end

@testset "higher-order conditionals are normalized mixed derivatives" begin
    C = ClaytonCopula{3}(1.5)
    fixed = [0.38, 0.47]
    target = 0.64
    D = condition(C, (1, 2), Tuple(fixed))
    numerator = ForwardDiff.hessian(
        x -> cdf(C, [x[1], x[2], target]), fixed)[1, 2]
    normalizer = ForwardDiff.hessian(
        x -> cdf(C, [x[1], x[2], 1.0]), fixed)[1, 2]
    @test cdf(D, target) ≈ numerator / normalizer atol=3e-5 rtol=3e-5

    h = 1e-5
    conditional_derivative = (cdf(D, target + h) - cdf(D, target - h)) / (2h)
    @test pdf(D, target) ≈ conditional_derivative atol=3e-5 rtol=3e-5

    ρ = 0.3
    gaussian = GaussianCopula{3}(ρ)
    joint = condition(gaussian, 1, 0.41)
    point = [0.57, 0.69]
    # Conditioning an exchangeable Gaussian correlation matrix on one
    # coordinate leaves correlation (ρ-ρ²)/(1-ρ²)=ρ/(1+ρ).
    expected = GaussianCopula{2}(ρ / (1 + ρ))
    @test joint.C.Σ ≈ expected.Σ atol=2e-12 rtol=2e-12
end

@testset "Rosenblatt coordinates are conditional distribution functions" begin
    C = GaussianCopula{3}(0.3)
    u = [0.31, 0.52, 0.74]
    R = rosenblatt(C, u)
    @test R[1] ≈ u[1]
    @test R[2] ≈ cdf(condition(C, 1, u[1]).m[1], u[2])
    @test R[3] ≈ cdf(condition(C, (1, 2), (u[1], u[2])), u[3])
    @test inverse_rosenblatt(C, R) ≈ u atol=2e-6 rtol=2e-6

    independent = IndependentCopula{3}()
    @test rosenblatt(independent, u) == u
    @test inverse_rosenblatt(independent, u) == u
end

@testset "Rosenblatt conditional densities factorize the copula density" begin
    u = [0.31, 0.52, 0.74]
    for C in (ClaytonCopula{3}(1.5), GaussianCopula{3}(0.3))
        second = condition(C, (1,), (u[1],))
        third = condition(C, (1, 2), (u[1], u[2]))
        second_density = pdf(second, u[2:3])
        third_density = pdf(third, u[3])
        marginal_second = pdf(second.m[1], u[2])
        @test pdf(C, u) ≈ marginal_second * third_density
        @test second_density ≈ marginal_second * third_density
    end
end

@testset "conditional densities are normalized" begin
    for D in (condition(ClaytonCopula{3}(1.5), (1, 2), (0.38, 0.47)),
              condition(GalambosCopula{2}(1.0), 1, 0.41))
        mass, _ = QuadGK.quadgk(x -> pdf(D, x), 0.0, 1.0; rtol=2e-6)
        @test mass ≈ 1 atol=2e-5
    end
end

@testset "generator transform representations" begin
    for G in (Copulas.ClaytonGenerator(1.5),)
        frailty = Copulas.frailty(G)
        for t in (0.2, 0.7, 1.4)
            @test Copulas.ϕ(G, t) ≈ Distributions.mgf(frailty, -t) atol=2e-10
        end
    end

    radial = Gamma(2.5, 0.8)
    order = 3.5
    G = WilliamsonGenerator(radial, order)
    for t in (0.2, 1.4)
        expected = Distributions.expectation(radial) do r
            r > t ? (1 - t / r)^(order - 1) : 0.0
        end
        @test Copulas.ϕ(G, t) ≈ expected
    end

    reduced_order = 2.25
    reduced_radial = Copulas.𝒲₋₁(G, reduced_order)
    reconstructed = WilliamsonGenerator(reduced_radial, reduced_order)
    for t in (0.2, 1.4)
        @test Copulas.ϕ(reconstructed, t) ≈ Copulas.ϕ(G, t) atol=2e-7 rtol=2e-7
    end
end

@testset "generator monotonicity signs" begin
    for G in (Copulas.ClaytonGenerator(1.5),
              WilliamsonGenerator(Gamma(2.5, 0.8), 3.5))
        for t in (0.2, 0.7, 1.4), k in 0:2
            @test (-1)^k * Copulas.ϕ⁽ᵏ⁾(G, k, t) >= -1e-10
        end
    end
end

@testset "stable-tail convexity" begin
    for (tail, d) in TAIL_CASES
        x = collect(range(0.25, 0.85; length=d))
        y = reverse(x) .+ 0.17
        λ = 0.37
        @test Copulas.ℓ(tail, λ .* x .+ (1 - λ) .* y) <=
              λ * Copulas.ℓ(tail, x) + (1 - λ) * Copulas.ℓ(tail, y) + 2e-6
    end
end

@testset "Archimax limiting constructions and dependence" begin
    C = ArchimaxCopula{2}(
        Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))
    u = [0.37, 0.68]
    archimedean = ClaytonCopula{2}(1.5)
    @test cdf(ArchimaxCopula{2}(archimedean.G, Copulas.NoTail()), u) ≈
          cdf(archimedean, u)

    ev = GalambosCopula{2}(1.0)
    @test cdf(ArchimaxCopula{2}(Copulas.IndependentGenerator(), ev.tail), u) ≈
          cdf(ev, u)

    τ_tail = Copulas.τ(ExtremeValueCopula{2}(C.tail))
    τ_generator = Copulas.τ(C.gen)
    @test Copulas.τ(C) ≈ τ_tail + (1 - τ_tail) * τ_generator
end

@testset "multivariate Archimedean defining formula" begin
    u = [0.32, 0.54, 0.76]
    for C in (ClaytonCopula{3}(1.5), FrankCopula{3}(2.0),
              GumbelCopula{3}(1.5))
        @test cdf(C, u) ≈ Copulas.ϕ(C.G,
            sum(Copulas.ϕ⁻¹.(Ref(C.G), u)))
        prove_dispatch_route!(:cdf, C, :archimedean_defining_formula)
    end
end

@testset "multivariate Gaussian CDF agrees with density integration" begin
    C = GaussianCopula{3}(0.3)
    u = [0.32, 0.54, 0.76]
    expected = invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
    @test cdf(C, u) ≈ expected atol=1e-3 rtol=1e-3
    prove_dispatch_route!(:cdf, C, :density_integration)
end

@testset "survival transformation is an involution" begin
    C = ClaytonCopula{3}(1.5)
    flips = (1, 3)
    restored = SurvivalCopula{3}(SurvivalCopula{3}(C, flips), flips)
    u = [0.32, 0.54, 0.76]
    @test cdf(restored, u) ≈ cdf(C, u)
    @test pdf(restored, u) ≈ pdf(C, u)
    wrapped = SurvivalCopula{3}(C, flips)
    expected = 0.0
    for mask in Iterators.product((0:1 for _ in flips)...)
        point = copy(u)
        for i in flips
            point[i] = 1.0
        end
        for (k, i) in pairs(flips)
            mask[k] == 1 && (point[i] = 1 - u[i])
        end
        expected += (-1)^sum(mask) * cdf(C, point)
    end
    @test cdf(wrapped, u) ≈ expected
    prove_dispatch_route!(:cdf, wrapped, :survival_inclusion_exclusion)
end

@testset "dependence measures agree with their definitions" begin
    C = FGMCopula{2}(0.4)
    integral, _ = HCubature.hcubature(u -> cdf(C, u), zeros(2), ones(2);
                                      rtol=2e-5)
    @test Copulas.ρ(C) ≈ 12integral - 3 atol=2e-4
    @test Copulas.β(C) ≈ 4cdf(C, [0.5, 0.5]) - 1

    @test Copulas.τ(IndependentCopula{2}()) == 0
    @test Copulas.ρ(IndependentCopula{2}()) == 0
    @test Copulas.β(IndependentCopula{2}()) == 0
    @test Copulas.γ(IndependentCopula{2}()) == 0
    @test Copulas.τ(MCopula{2}()) == 1
    @test Copulas.ρ(MCopula{2}()) == 1
    @test Copulas.τ(WCopula{2}()) == -1
    @test Copulas.ρ(WCopula{2}()) == -1

    for C in (IndependentCopula{2}(), IndependentCopula{3}(),
              MCopula{2}(), MCopula{3}(), WCopula{2}())
        for measure in SCALAR_DEPENDENCE_MEASURES
            if applicable(measure, C) &&
               !(measure in (Copulas.ι,) && C isa WCopula)
                value = measure(C)
                @test value isa Real
                prove_dependence_route!(measure, C)
            end
        end
    end
end

@testset "singular and mixed copulas use mass identities" begin
    u = [0.37, 0.68]
    @test cdf(MCopula{2}(), u) == minimum(u)
    @test cdf(WCopula{2}(), u) == max(sum(u) - 1, 0)

    lower = [0.2, 0.2]
    upper = [0.7, 0.7]
    @test Copulas.measure(MCopula{2}(), lower, upper) ≈ 0.5
    @test Copulas.measure(WCopula{2}(), lower, upper) ≈ 0.4

    C = MOCopula{2}(0.2, 0.3, 0.4)
    split = 0.45
    whole = Copulas.measure(C, [0.1, 0.15], [0.8, 0.75])
    left = Copulas.measure(C, [0.1, 0.15], [split, 0.75])
    right = Copulas.measure(C, [split, 0.15], [0.8, 0.75])
    @test whole ≈ left + right atol=1e-12

    # Generalized conditional quantiles remain valid in the presence of atoms;
    # a bijective Rosenblatt identity is intentionally not asserted here.
    D = condition(C, 1, 0.4)
    probabilities = collect(0.05:0.05:0.95)
    quantiles = quantile.(Ref(D), probabilities)
    @test issorted(quantiles)
    @test any(iszero, diff(quantiles))
    for (p, q) in zip(probabilities, quantiles)
        @test cdf(D, q) >= p - 1e-10
    end
end
