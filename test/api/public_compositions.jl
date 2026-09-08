# Public-API proof: checks public constructors whose behavior is chiefly
# composition or specialization into generators, tails, and complete copulas.
@testset "remaining public generator constructors" begin
    @test Copulas.τ(Copulas.IndependentGenerator()) == 0
    @test ArchimedeanCopula{3}(Copulas.IndependentGenerator()) isa IndependentCopula{3}

    frailty_generator = Copulas.FrailtyGenerator(Exponential())
    @test params(frailty_generator) == (F=Exponential(),)
    for t in (0.0, 0.5, 1.0)
        @test Copulas.ϕ(frailty_generator, t) == mgf(Exponential(), -t)
    end

    empirical = EmpiricalGenerator(_FIXTURE_DATA)
    @test empirical isa Copulas.Generator
    @test 0 <= Copulas.ϕ(empirical, 0.5) <= 1
    ranked_empirical = EmpiricalGenerator(_FIXTURE_DATA; pseudo_values=false)
    @test params(ranked_empirical) == params(EmpiricalGenerator(pseudos(_FIXTURE_DATA)))
end

@testset "Williamson inverse public distribution" begin
    G = Copulas.ClaytonGenerator(1.0)
    @test Copulas.𝒲(Dirac(1.0), 2.0) isa WilliamsonGenerator
    for order in (2, 2.5)
        radial = Copulas.𝒲₋₁(G, order)
        @test minimum(radial) >= 0
        @test cdf(radial, minimum(radial)) >= 0
        @test pdf(radial, 0.7) >= 0
        @test logpdf(radial, 0.7) ≈ log(pdf(radial, 0.7))
        @test maximum(radial) >= minimum(radial)
        @test quantile(radial, 0.5) >= minimum(radial)
        @test rand(StableRNG(81), radial) >= minimum(radial)
    end

    source = Copulas.𝒲(LogNormal(), 3.0)
    reduced = Copulas.𝒲₋₁(source, 2.5)
    restored = Copulas.𝒲(reduced, 2.5)
    @test Copulas.max_monotony(restored) == 3.0
    @test Copulas.ϕ(restored, 0.7) ≈ Copulas.ϕ(source, 0.7)
end

@testset "discrete spectral public API" begin
    B = [0.7 0.3; 0.2 0.8]
    tail = Copulas.DiscreteSpectralTail(B)
    C = ExtremeValueCopula{2}(tail)
    @test params(tail) == (B=Float64.(B),)
    @test Copulas.ℓ(tail, [1.0, 0.0]) ≈ 1
    @test length(C) == 2
    @test size(rand(StableRNG(82), C, 3)) == (2, 3)
    @test_throws ArgumentError logpdf(C, [0.4, 0.7])
end
