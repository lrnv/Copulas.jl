# Specialization-equivalence layer: conditional-distribution and distortion
# fast paths are checked against inversion identities, generic conditionals,
# log-scale definitions, or independent Gaussian conditioning algebra.

@testset "Gaussian distortion log-scale formulas" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))
    N = Normal()
    for u in (1e-12, 0.2, 0.5, 0.8)
        q = quantile(N, u)
        z = (q - D.μz) / D.σz
        reference = logpdf(N, z) - log(abs(D.σz)) - logpdf(N, q)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 1e-13
        @test logpdf(D, u) ≈ reference atol = 1e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
    @test logpdf(D, -0.1) == -Inf
end

@testset "Student distortion logcdf" begin
    D = condition(TCopula{2}(4, [1.0 0.5; 0.5 1.0]), (1,), (0.3,))
    @test D.Tu isa TDist
    @test D.Tcond isa TDist
    for u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "Elliptical conditioning shares matrix factorizations" begin
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    for C in (GaussianCopula{3}(Σ), TCopula{3}(4, Σ))
        conditioned = condition(C, (1,), (0.35,))
        @test length(conditioned.m) == 2
        for (k, i) in enumerate((2, 3)), u in (0.2, 0.7)
            reference = Copulas.DistortionFromCop(C, (1,), (0.35,), i)
            @test cdf(conditioned.m[k], u) ≈ cdf(reference, u) atol = 2e-12
        end
    end
end

@testset "Distorted distribution logcdf" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))(Logistic())
    @test D isa Copulas.DistortedDist
    for x in (-8.0, -0.5, 1.0)
        @test logcdf(D, x) ≈ logcdf(D.D, cdf(D.X, x)) atol = 2e-13
    end
end

@testset "Archimedean distortion logcdf" begin
    distortions = (
        condition(ClaytonCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(FrankCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(GumbelCopula{3}(2.0), (1, 2), (0.3, 0.6)),
    )
    for D in distortions, u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 3e-12
    end
    @test all(logcdf(D, 0.0) == -Inf for D in distortions)
    @test all(logcdf(D, 1.0) == 0.0 for D in distortions)
end

@testset "Flip distortion logcdf" begin
    S = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    D = condition(S, (1,), (0.3,))
    @test D isa Copulas.FlipDistortion
    for u in (0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-12
    end
    u = 1e-12
    @test logcdf(D, u) ≈ LogExpFunctions.log1mexp(logcdf(D.base, 1 - u)) atol = 2e-12
    @test isfinite(logcdf(D, u))
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "FGM distortion log-scale formulas" begin
    for θ in (-0.8, 0.8), uⱼ in (0.2, 0.7)
        D = condition(FGMCopula{2}(θ), (1,), (uⱼ,))
        for u in (1e-12, 0.2, 0.5, 0.8)
            @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-14
        end
        @test logcdf(D, 0.0) == -Inf
        @test logcdf(D, 1.0) == 0.0
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Generic ConditionalCopula density" begin
    C = GaussianCopula{3}([
        1.0 0.35 0.20
        0.35 1.0 0.25
        0.20 0.25 1.0
    ])
    js = (3,)
    ujs = (0.4,)
    generic = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, ujs)
    Cgeneric = FGMCopula{3}([0.1, 0.2, 0.3, 0.4])
    conditioned = condition(Cgeneric, js, ujs)
    @test conditioned.C isa Copulas.ConditionalCopula
    @test conditioned.m === conditioned.C.distortions
    @test conditioned.C.is == (1, 2)
    @test generic.logden == log(generic.den)
    specialized = Copulas.ConditionalCopula(C, js, ujs)

    for u in ([0.25, 0.35], [0.5, 0.5], [0.75, 0.65])
        @test isapprox(logpdf(generic, u), logpdf(specialized, u); atol=1e-8, rtol=1e-8)
        @test isapprox(pdf(generic, u), pdf(specialized, u); atol=1e-8, rtol=1e-8)
    end
    @test pdf(generic, [-0.1, 0.5]) == 0

    Cclayton = ClaytonCopula{3}(2.0)
    generic_big = @invoke Copulas.ConditionalCopula(
        Cclayton::Copulas.Copula{3},
        (3,),
        (big"0.4",),
    )
    value_big = logpdf(generic_big, BigFloat[0.35, 0.65])
    @test value_big isa BigFloat
    @test isfinite(value_big)
end
