@testset "Liebscher template MLE" begin
    Ctrue = LiebscherCopula(
        2,
        (ClaytonCopula(2, 1.5), GumbelCopula(2, 1.8)),
        [0.3 0.7; 0.7 0.3],
    )
    C0 = LiebscherCopula(
        2,
        (ClaytonCopula(2, 0.8), GumbelCopula(2, 1.2)),
        [0.5 0.5; 0.5 0.5],
    )

    p0 = Copulas.Paramorph.param_space(C0)
    @test Copulas.Paramorph.names(p0) == (:C1_θ, :C2_θ, :a1, :a2)
    @test Copulas.Paramorph.dimension(p0) == 4

    U = rand(StableRNG(20_120), Ctrue, 300)
    fitted = fit(C0, U)

    @test fitted isa LiebscherCopula{2}
    @test typeof(fitted.copulas) == typeof(C0.copulas)
    @test loglikelihood(fitted, U) >= loglikelihood(C0, U)
    @test all(fitted.weights .> 0)
    @test all(vec(sum(fitted.weights; dims=1)) .≈ 1)

    M = fit(CopulaModel, C0, U)
    @test fitted_distribution(M) isa LiebscherCopula{2}
    @test coefnames(M) == ["C1_θ", "C2_θ", "a1_1", "a2_1", "a1_2", "a2_2"]
    @test length(coef(M)) == 6
    @test dof(M) == 4
    @test Copulas.Paramorph.dimension(Copulas.Paramorph.param_space(fitted_distribution(M))) == 4

    @test Copulas._available_fitting_methods(typeof(C0), 2) == ()
    @test_throws ArgumentError fit(typeof(C0), U)
    @test_throws ArgumentError fit(C0, U; method=:itau)
end

@testset "Liebscher template MLE preserves structural zero weights" begin
    Ctrue = LiebscherCopula(
        2,
        (ClaytonCopula(2, 1.5), GumbelCopula(2, 1.8)),
        [0.0 0.7; 1.0 0.3],
    )
    C0 = LiebscherCopula(
        2,
        (ClaytonCopula(2, 0.8), GumbelCopula(2, 1.2)),
        [0.0 0.5; 1.0 0.5],
    )

    @test Copulas.Paramorph.dimension(Copulas.Paramorph.param_space(C0)) == 3

    U = rand(StableRNG(20_121), Ctrue, 200)
    fitted = fit(C0, U)

    @test fitted.weights[1, 1] == 0
    @test fitted.weights[2, 1] == 1
    @test fitted.weights[1, 2] > 0
    @test fitted.weights[2, 2] > 0
    @test sum(fitted.weights[:, 2]) ≈ 1
    @test Copulas.Paramorph.dimension(Copulas.Paramorph.param_space(fitted)) == 3
end
