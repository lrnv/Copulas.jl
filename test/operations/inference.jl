@testset "post-fit inference contract" begin
    U = rand(StableRNG(48_100), GumbelCopula{2}(2.0), 80)

    mle_model = fit(CopulaModel, GumbelCopula{2}, U;
                    method=:mle)
    Ih = infer(mle_model)
    @test Ih isa Copulas.CopulaInference
    @test Ih.method === :hessian
    @test size(vcov(Ih)) == (dof(mle_model), dof(mle_model))
    @test all(isfinite, vcov(Ih))
    @test length(stderror(Ih)) == dof(mle_model)
    lower, upper = confint(Ih)
    @test all(lower .< coef(mle_model))
    @test all(coef(mle_model) .< upper)
    @test Ih.method === :hessian

    rank_model = fit(CopulaModel, GumbelCopula{2}, U;
                     method=:itau)
    Ig = infer(rank_model; rng=StableRNG(48_101), nresamples=20)
    Ig_repeat = infer(rank_model; rng=StableRNG(48_101), nresamples=20)
    @test Ig.method === :godambe
    @test size(vcov(Ig)) == (1, 1)
    @test vcov(Ig) == vcov(Ig_repeat)

    U3 = rand(StableRNG(48_102), GaussianCopula{3}(0.35), 120)
    rank_model_3d = fit(CopulaModel, GaussianCopula, U3; method=:itau)
    Ig3 = infer(rank_model_3d; rng=StableRNG(48_103), nresamples=20)
    Ig3_repeat = infer(rank_model_3d; rng=StableRNG(48_103), nresamples=20)
    @test Ig3.method === :godambe_pairwise
    @test size(vcov(Ig3)) == (3, 3)
    @test all(isfinite, vcov(Ig3))
    @test vcov(Ig3) == vcov(Ig3_repeat)
    @test_throws ArgumentError infer(rank_model_3d; method=:godambe, rng=StableRNG(48_104), nresamples=20)
    @test_throws ArgumentError infer(rank_model_3d; method=:godambe_pairwise, rng=StableRNG(48_105), nresamples=1)
    X = [2.0 8.0 1.0 5.0 3.0 7.0; 4.0 1.0 6.0 2.0 5.0 3.0]
    mpl_model = fit(CopulaModel, ClaytonCopula{2}, X; method=:mpl, pseudo_values=false)
    @test Copulas.fitting_method(Copulas._refit(mpl_model, X; replay_input=true)) === :mpl
    @test_throws ArgumentError infer(mpl_model)
    Ib = infer(mpl_model; method=:bootstrap, nresamples=3, rng=StableRNG(48_106))
    @test Ib.method === :bootstrap
    @test_throws ArgumentError infer(mpl_model; method=:bootstrap, nresamples=1)
    @test fitted_distribution(mle_model) === fitted_distribution(Ih.model)
    @test Ih !== Ig
end
