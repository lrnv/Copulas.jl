# Mathematical-path layer: expensive CDF/PDF, derivative, integral, rectangle,
# and transform equivalences are checked once per implementation mechanism,
# not for every parameterization of every public family.
const DENSITY_COHERENCE_CASES = (
    ClaytonCopula{2}(1.5),
    GaussianCopula{2}(0.3),
    GalambosCopula{2}(1.0),
    ArchimaxCopula{2}(Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
    FGMCopula{2}(0.4),
    LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
)

const CDF_DERIVATIVE_CASES = DENSITY_COHERENCE_CASES[1:5]

# Classification inherited from the former generic suite:
# - universal invariants: copula margins, support and API identities live in
#   contracts/copulas.jl;
# - mechanism identities: derivatives, integrals, transforms and defining
#   representations are checked below on one representative implementation;
# - family formulas, limits and fixed regressions remain in focused old tests
#   until the family-regression migration phase.

# Smooth polynomial oracle. Its closed forms are independent of the generic
# integration, conditioning and Rosenblatt machinery exercised below.
struct PolynomialOracleCopula{T} <: Copulas.Copula{2}
    θ::T
end
Distributions.params(C::PolynomialOracleCopula) = (; θ=C.θ)
function Copulas._cdf(C::PolynomialOracleCopula, u)
    x, y = u
    return x * y * (1 + C.θ * (1 - x) * (1 - y))
end
function Distributions._logpdf(C::PolynomialOracleCopula, u)
    x, y = u
    return log1p(C.θ * (1 - 2x) * (1 - 2y))
end
_oracle_cdf(C::PolynomialOracleCopula, u) =
    u[1] * u[2] * (1 + C.θ * (1 - u[1]) * (1 - u[2]))
_oracle_pdf(C::PolynomialOracleCopula, u) =
    1 + C.θ * (1 - 2u[1]) * (1 - 2u[2])
_oracle_conditional_cdf(C::PolynomialOracleCopula, conditioned, target) =
    target * (1 + C.θ * (1 - 2conditioned) * (1 - target))

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

@testset "generic smooth-copula oracle" begin
    C = PolynomialOracleCopula(0.4)
    u = [0.37, 0.68]
    @test cdf(C, u) ≈ _oracle_cdf(C, u)
    @test pdf(C, u) ≈ _oracle_pdf(C, u)

    # Bypass the analytic _cdf method and exercise Copula.jl's density integral.
    integrated = invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
    @test integrated ≈ _oracle_cdf(C, u) atol=2e-5

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
end

@testset "generic tail and extreme-value oracle" begin
    tail = LogisticOracleTail(1.5)
    x = [0.4, 0.7]
    expected_ℓ = sum(x .^ tail.θ)^(inv(tail.θ))
    @test Copulas.ℓ(tail, x) ≈ expected_ℓ
    @test Copulas.A(tail, Tuple(x ./ sum(x))) ≈ expected_ℓ / sum(x)

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
end

@testset "generic Williamson oracle" begin
    radial = Uniform(1.0, 2.0)
    G = WilliamsonGenerator(radial, 3.0)
    t = 0.4
    expected = 1 - 2t * log(2) + t^2 / 2
    @test Copulas.ϕ(G, t) ≈ expected
    @test Copulas.𝒲₋₁(G, 3.0) === radial
end

@testset "CDF and density mathematical coherence" begin
    for C in DENSITY_COHERENCE_CASES
        @testset "$(nameof(typeof(C)))" begin
            total, _ = HCubature.hcubature(u -> pdf(C, u), zeros(2), ones(2);
                                            rtol=2e-3)
            @test total ≈ 1 atol=5e-3

            upper = [0.55, 0.65]
            partial, _ = HCubature.hcubature(u -> pdf(C, u), zeros(2), upper;
                                              rtol=2e-3)
            @test partial ≈ cdf(C, upper) atol=5e-3

            lower = [0.15, 0.25]
            rectangle, _ = HCubature.hcubature(u -> pdf(C, u), lower, upper;
                                                rtol=2e-3)
            @test rectangle ≈ Copulas.measure(C, lower, upper) atol=5e-3
        end
    end
end

@testset "density is the mixed CDF derivative" begin
    for C in CDF_DERIVATIVE_CASES
        u = [0.43, 0.61]
        derivative = ForwardDiff.hessian(x -> cdf(C, x), u)[1, 2]
        @test pdf(C, u) ≈ derivative atol=2e-4 rtol=2e-3
    end
end

@testset "conditional CDF is the normalized CDF derivative" begin
    for C in (ClaytonCopula{2}(1.5), GaussianCopula{2}(0.3),
              GalambosCopula{2}(1.0), FGMCopula{2}(0.4))
        conditioned = 0.41
        target = 0.63
        D = condition(C, 1, conditioned)
        derivative = ForwardDiff.derivative(v -> cdf(C, [v, target]), conditioned)
        @test cdf(D, target) ≈ derivative atol=2e-5 rtol=2e-5
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
    C = GaussianCopula{3}(0.3)
    lower = [0.12, 0.18, 0.24]
    upper = [0.68, 0.73, 0.81]
    expected = sum(Iterators.product((0:1 for _ in 1:3)...)) do corner
        point = [corner[i] == 1 ? upper[i] : lower[i] for i in 1:3]
        (-1)^(3 - sum(corner)) * cdf(C, point)
    end
    @test Copulas.measure(C, lower, upper) ≈ expected atol=1e-12

    split = 0.46
    left_upper = copy(upper)
    left_upper[1] = split
    right_lower = copy(lower)
    right_lower[1] = split
    @test Copulas.measure(C, lower, upper) ≈
          Copulas.measure(C, lower, left_upper) +
          Copulas.measure(C, right_lower, upper) atol=1e-12
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
        second = condition(C, 1, u[1])
        third = condition(C, (1, 2), (u[1], u[2]))
        @test pdf(C, u) ≈ pdf(second, u[2]) * pdf(third, u[3])
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
    for G in (Copulas.ClaytonGenerator(1.5), Copulas.FrankGenerator(2.0))
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
