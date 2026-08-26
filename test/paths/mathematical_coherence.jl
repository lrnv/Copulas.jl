# Expensive mathematical equivalences are checked once per implementation
# mechanism, not for every parameterization of every public family.
const DENSITY_COHERENCE_CASES = (
    ClaytonCopula{2}(1.5),
    GaussianCopula{2}(0.3),
    GalambosCopula{2}(1.0),
    ArchimaxCopula{2}(Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
    FGMCopula{2}(0.4),
    LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
)

const CDF_DERIVATIVE_CASES = DENSITY_COHERENCE_CASES[1:5]

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
