@testset "conditioning fast paths are numeric-generic" begin
    for T in (Float32, BigFloat)
        # Keep the BigFloat Gaussian inputs on the binary Float64-derived 0.8:
        # exact decimal BigFloat inputs can make SpecialFunctions.erfcinv fail
        # to terminate; see https://github.com/JuliaMath/SpecialFunctions.jl/issues/557.
        v = T === BigFloat ? big(0.8) : T(0.41)
        w = T === BigFloat ? big(0.8) : T(0.53)

        gaussian = GaussianCopula{3}(T[1 0.3 0.2; 0.3 1 0.25; 0.2 0.25 1])
        GD = condition(gaussian, (1, 2), (v, w))
        @test GD isa Copulas.GaussianDistortion
        GC = condition(gaussian, (1,), (v,))
        @test GC isa SklarDist
        @test GC.C isa GaussianCopula{2}
        @test all(m -> m isa Copulas.GaussianDistortion, GC.m)
        if T === BigFloat
            @test GD.μz isa BigFloat
            @test GD.σz isa BigFloat
            @test eltype(GC.C) == BigFloat
            @test all(m -> m.μz isa BigFloat && m.σz isa BigFloat, GC.m)
        end

        clayton = ClaytonCopula{3}(T(1.5))
        AD = condition(clayton, (1, 2), (v, T(0.53)))
        @test AD isa Copulas.ArchimedeanDistortion
        @test AD.sJ isa T
        @test AD.den isa T
        AC = condition(clayton, (1,), (v,))
        @test AC isa SklarDist
        @test AC.C isa ArchimedeanCopula{2}
        @test eltype(AC.C) == T
        @test AC.C.G isa Copulas.TiltedGenerator
        @test AC.C.G.sJ isa T

        @test condition(IndependentCopula{2}(), (1,), (v,)) isa Uniform
    end

    # Student conditioning follows the same analytical path for non-Float64
    # inputs supported by Distributions/StatsFuns. The Student quantile backend
    # currently promotes Float32 and does not support BigFloat.
    T = Float32
    v = T(0.41)
    student = TCopula{3}(T(5), T[1 0.3 0.2; 0.3 1 0.25; 0.2 0.25 1])
    TD = condition(student, (1, 2), (v, T(0.53)))
    @test TD isa Copulas.StudentDistortion
    TC = condition(student, (1,), (v,))
    @test TC isa SklarDist
    @test TC.C isa TCopula{2}
    @test all(m -> m isa Copulas.StudentDistortion, TC.m)

    data = Float32[0.1 0.3 0.6 0.9; 0.2 0.8 0.4 0.7]
    beta = BetaCopula(data)
    BD = condition(beta, (1,), (Float32(0.4),))
    @test BD isa Distributions.MixtureModel
    @test !(BD isa Copulas.DistortionFromCop)

    checker = CheckerboardCopula(data; m=2)
    HD = condition(checker, (1,), (Float32(0.4),))
    @test HD isa Copulas.HistogramBinDistortion

    reflected = SurvivalCopula(ClaytonCopula{2}(Float32(1.5)), (1,))
    RD = condition(reflected, (1,), (Float32(0.4),))
    @test RD isa Copulas.FlipDistortion
end