# Mathematical-path layer: expensive CDF/PDF, derivative, integral, rectangle,
# and transform equivalences are checked once per implementation mechanism,
# not for every parameterization of every public family.
# Classification inherited from the former generic suite:
# - universal invariants: copula margins, support and API identities live in
#   contracts/copulas.jl;
# - mechanism identities: derivatives, integrals, transforms and defining
#   representations are checked below on one representative implementation;
# - family formulas, limits and fixed regressions remain in focused old tests
#   until the family-regression migration phase.

# Smooth polynomial oracle. Its closed forms are independent of the generic
# integration, conditioning and Rosenblatt machinery exercised below.
struct PolynomialOracleCopula{d,T} <: Copulas.Copula{d}
    θ::T
end
PolynomialOracleCopula(θ) = PolynomialOracleCopula{2,typeof(θ)}(θ)
Distributions.params(C::PolynomialOracleCopula) = (; θ=C.θ)
function Copulas._cdf(C::PolynomialOracleCopula, u)
    return prod(u) * (1 + C.θ * prod(1 .- u))
end
function Distributions._logpdf(C::PolynomialOracleCopula, u)
    return log1p(C.θ * prod(1 .- 2 .* u))
end
_oracle_cdf(C::PolynomialOracleCopula, u) =
    prod(u) * (1 + C.θ * prod(1 .- u))
_oracle_pdf(C::PolynomialOracleCopula, u) =
    1 + C.θ * prod(1 .- 2 .* u)
_oracle_conditional_cdf(C::PolynomialOracleCopula, conditioned, target) =
    target * (1 + C.θ * (1 - 2conditioned) * (1 - target))

function Distributions._rand!(rng::Distributions.AbstractRNG,
                              C::PolynomialOracleCopula{2},
                              U::AbstractMatrix{T}) where {T<:Real}
    for j in axes(U, 2)
        x, p = rand(rng), rand(rng)
        y = Roots.find_zero(
            target -> _oracle_conditional_cdf(C, x, target) - p,
            (zero(T), one(T)), Roots.Bisection())
        U[1, j] = x
        U[2, j] = y
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
struct LogisticOracleTail{T} <: Copulas.Tail
    θ::T
end
Distributions.params(tail::LogisticOracleTail) = (; θ=tail.θ)
Copulas.ℓ(tail::LogisticOracleTail, x) =
    sum(xᵢ -> xᵢ^tail.θ, x)^(inv(tail.θ))
Copulas.A(tail::LogisticOracleTail, t::Real) =
    Copulas.ℓ(tail, (t, 1 - t))

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
        for x in (0.4, 1.2, 3.0)
            @test cdf(inverse, x) ≈ cdf(reference, x) atol=2e-7
            @test pdf(inverse, x) ≈ pdf(reference, x) atol=2e-7
        end
        for p in (0.2, 0.6, 0.9)
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
    @test Copulas.measure(IndependentCopula{3}(), lower, upper) ≈
          prod(upper - lower)
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

    gaussian = GaussianCopula{3}(0.3)
    joint = condition(gaussian, 1, 0.41)
    point = [0.57, 0.69]
    expected = (cdf(gaussian, [0.41 + h, point[1], point[2]]) -
                cdf(gaussian, [0.41 - h, point[1], point[2]])) / (2h)
    @test cdf(joint, point) ≈ expected atol=3e-5 rtol=3e-5
end

@testset "Rosenblatt coordinates are conditional distribution functions" begin
    C = GaussianCopula{3}(0.3)
    u = [0.31, 0.52, 0.74]
    R = rosenblatt(C, u)
    @test R[1] ≈ u[1]
    @test R[2] ≈ cdf(condition(C, 1, u[1]), u[2])
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
        marginal_second = pdf(condition(C, 1, u[1]), u[2])
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
    for t in (0.2, 0.7, 1.4)
        expected = Distributions.expectation(radial) do r
            r > t ? (1 - t / r)^(order - 1) : 0.0
        end
        @test Copulas.ϕ(G, t) ≈ expected
    end

    reduced_order = 2.25
    reduced_radial = Copulas.𝒲₋₁(G, reduced_order)
    reconstructed = WilliamsonGenerator(reduced_radial, reduced_order)
    for t in (0.2, 0.7, 1.4)
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
    C = ClaytonCopula{3}(1.5)
    u = [0.32, 0.54, 0.76]
    @test cdf(C, u) ≈ Copulas.ϕ(C.G, sum(Copulas.ϕ⁻¹.(Ref(C.G), u)))
end

@testset "survival transformation is an involution" begin
    C = ClaytonCopula{3}(1.5)
    flips = (1, 3)
    restored = SurvivalCopula{3}(SurvivalCopula{3}(C, flips), flips)
    u = [0.32, 0.54, 0.76]
    @test cdf(restored, u) ≈ cdf(C, u)
    @test pdf(restored, u) ≈ pdf(C, u)
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
