@testset "post-fit inference contract" begin
    U = rand(StableRNG(48_100), GumbelCopula{2}(2.0), 80)

    mle_model = fit(CopulaModel, GumbelCopula{2}, U;
                    method=:mle, derived_measures=false)
    Ih = infer(mle_model)
    @test Ih isa CopulaInference
    @test Ih.method === :hessian
    @test size(vcov(Ih)) == (dof(mle_model), dof(mle_model))
    @test all(isfinite, vcov(Ih))
    @test length(stderror(Ih)) == dof(mle_model)
    lower, upper = confint(Ih)
    @test all(lower .< coef(mle_model))
    @test all(coef(mle_model) .< upper)
    @test inference_diagnostics(Ih).method === :hessian

    rank_model = fit(CopulaModel, GumbelCopula{2}, U;
                     method=:itau, derived_measures=false)
    Ig = infer(rank_model)
    @test Ig.method === :godambe
    @test size(vcov(Ig)) == (1, 1)

    X = [2.0 8.0 1.0 5.0 3.0 7.0;
         4.0 1.0 6.0 2.0 5.0 3.0]
    mpl_model = fit(CopulaModel, ClaytonCopula{2}, X;
                    method=:mpl, pseudo_values=false,
                    derived_measures=false)
    @test Copulas._refit(mpl_model, X; replay_input=true).method === :mpl
    @test_throws ArgumentError infer(mpl_model)
    Ib = infer(mpl_model; method=:bootstrap, nresamples=3,
               rng=StableRNG(48_101))
    @test Ib.method === :bootstrap
    @test inference_diagnostics(Ib).nresamples == 3
    @test_throws ArgumentError infer(mpl_model; method=:bootstrap,
                                     nresamples=1)

    @test fitteddistribution(mle_model) === fitteddistribution(Ih.model)
    @test Ih !== Ig
end
