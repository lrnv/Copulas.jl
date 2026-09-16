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

    # Analytical inference is deliberately strict: singular/indefinite
    # matrices fail instead of being silently ridge/eigenvalue regularized.
    @test Copulas._invert_observed_information([2.0 0.0; 0.0 4.0]) ≈
          [0.5 0.0; 0.0 0.25]
    @test_throws ArgumentError Copulas._invert_observed_information(
        [1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError Copulas._invert_observed_information(
        [1.0 2.0; 2.0 1.0])
    good_cov = [1.0 0.1; 0.1 2.0]
    @test Matrix(Copulas._validate_inference_covariance(good_cov)) ≈ good_cov
    @test_throws ArgumentError Copulas._validate_inference_covariance(
        [1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError Copulas._validate_inference_covariance(
        [1.0 2.0; 2.0 1.0])

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

@testset "inference on a weighted model" begin
    # A weight is "how many observations this column counts for", so every
    # procedure is the unweighted one on the sample in which observation j is
    # repeated weights[j] times: the Hessian of the weighted log-likelihood is
    # that sample's observed information, and the resampling methods draw
    # observation j with probability weights[j] / n, then work unweighted.
    n = 200
    counts = zeros(Int, n)
    for slot in rand(StableRNG(48_200), 1:n, n)
        counts[slot] += 1
    end
    kept = findall(>(0), counts)
    replicate(X) = hcat((repeat(X[:, j], 1, counts[j]) for j in kept)...)

    @testset "observed information: $(nameof(CT))" for (CT, C) in (
            (ClaytonCopula, ClaytonCopula{2}(2.0)),
            (GumbelCopula, GumbelCopula{3}(1.6)),
            (GaussianCopula, GaussianCopula([1.0 0.5; 0.5 1.0])))
        U = rand(StableRNG(48_201), C, n)
        unweighted = infer(fit(CopulaModel, CT, U))
        for weights in (ones(n), fill(2.5, n))
            weighted = infer(fit(CopulaModel, CT, U; weights))
            @test weighted.method === :hessian
            @test vcov(weighted) == vcov(unweighted)
        end
        replicated = infer(fit(CopulaModel, CT, replicate(U)))
        weighted = infer(fit(CopulaModel, CT, U; weights=counts))
        @test vcov(weighted) ≈ vcov(replicated) rtol=1e-6
        @test_throws ArgumentError infer(fit(CopulaModel, CT, U; weights=counts); method=:jackknife)
    end

    @testset "Godambe resamples the replicated sample" begin
        U = rand(StableRNG(48_202), GumbelCopula{2}(2.0), n)
        weighted = fit(CopulaModel, GumbelCopula, U; method=:itau, weights=counts)
        Ig = infer(weighted; rng=StableRNG(48_203), nresamples=20)
        @test Ig.method === :godambe
        @test vcov(Ig) == vcov(infer(weighted; rng=StableRNG(48_203), nresamples=20))
        @test all(isfinite, vcov(Ig))
        # The weighted sampler draws the same distribution as a uniform draw
        # from the replicated sample: the two bootstrap variances agree to
        # Monte Carlo resolution.
        B = 3000
        replicated = fit(CopulaModel, GumbelCopula, replicate(U); method=:itau)
        @test vcov(infer(weighted; rng=StableRNG(48_204), nresamples=B))[1] ≈
              vcov(infer(replicated; rng=StableRNG(48_205), nresamples=B))[1] rtol=0.1
        U3 = rand(StableRNG(48_206), GaussianCopula{3}(0.35), n)
        I3 = infer(fit(CopulaModel, GaussianCopula, U3; method=:itau, weights=counts);
                   rng=StableRNG(48_207), nresamples=20)
        @test I3.method === :godambe_pairwise
        @test size(vcov(I3)) == (3, 3) && all(isfinite, vcov(I3))
    end

    @testset "bootstrap refits each resample unweighted" begin
        X = randn(StableRNG(48_208), 2, n)
        weighted = fit(CopulaModel, ClaytonCopula, X; pseudo_values=false, weights=counts)
        @test Copulas._model_weights(Copulas._refit(weighted, X; replay_input=true)) === nothing
        @test Copulas.fitting_method(Copulas._refit(weighted, X; replay_input=true)) === :mpl
        Ib = infer(weighted; method=:bootstrap, nresamples=5, rng=StableRNG(48_209))
        @test Ib.method === :bootstrap
        @test vcov(Ib) == vcov(infer(weighted; method=:bootstrap, nresamples=5, rng=StableRNG(48_209)))
        @test_throws ArgumentError infer(weighted; method=:jackknife)
        # The Sklar route: margins, ranks and copula are refitted on each resample.
        S = SklarDist{ClaytonCopula,Tuple{Normal,Normal}}
        sklar = fit(CopulaModel, S, X; weights=counts)
        Is = infer(sklar; nresamples=3, rng=StableRNG(48_210))
        @test Is.method === :bootstrap
        @test size(vcov(Is)) == (dof(sklar), dof(sklar))
        @test_throws ArgumentError infer(sklar; method=:jackknife)
    end
end
