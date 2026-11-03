
# Joint maximum-likelihood fitting for Sklar distributions.
# Numerical comparisons use fixed random seeds and independent references.

# Joint parameter geometry for Sklar MLE.
# Tests round-trip reconstruction, coordinate dimensions,
# automatic differentiation, and finite-difference gradients.

# Internal marginal parameter-geometry contract for joint Sklar MLE.
# Each supported marginal must round-trip through unconstrained
# coordinates and permit likelihood differentiation.

@testset "Sklar marginal parameter geometry" begin

    cases = (
        (Normal, Normal(0.5, 1.3)),
        (Exponential, Exponential(2.0)),
        (Gamma, Gamma(2.5, 1.4)),
    )

    for (Family, original) in cases
        @testset "$Family" begin

            # Natural parameters -> free coordinates
            z = Copulas._sklar_margin_coordinates(original)

            @test length(z) == length(params(original))
            @test all(isfinite, z)

            # Free coordinates -> natural parameters
            recovered = Copulas._sklar_margin_from_coordinates(Family, z)

            @test recovered isa Family

            @test all(isapprox(a, b; rtol=1e-12, atol=1e-12) for (a, b) in zip(params(recovered), params(original)))

            # Differentiation of the marginal log-likelihood
            objective(v) = begin
                margin = Copulas._sklar_margin_from_coordinates(Family, v)
                return sum(logpdf(margin, x) for x in (1.1, 1.7))
            end

            gradient = ForwardDiff.gradient(objective, z)

            @test isfinite(objective(z))
            @test length(gradient) == length(z)
            @test all(isfinite, gradient)
        end
    end

    # Unsupported families must fail explicitly.
    @test_throws ArgumentError Copulas._sklar_margin_schema(Poisson)
end

@testset "Joint Sklar parameter geometry" begin

    cases = (
        (
            SklarDist(ClaytonCopula{2}(1.5), (Normal(0.3, 1.2), Exponential(1.6)),),
            [-0.4 0.6 1.5; 0.1 0.8 2.0], 4,),
        (
            SklarDist(IndependentCopula{2}(), (Normal(0.3, 1.2), Exponential(1.6)),),
            [-0.4 0.6 1.5; 0.1 0.8 2.0], 3,),
        (
            SklarDist(ClaytonCopula{3}(1.2), (Normal(), Exponential(), Gamma(2.5, 1.4)),),
            [
                -0.4 0.6 1.5
                 0.1 0.8 2.0
                 0.5 1.2 3.0
            ], 6,
        ),
    )

    for (S, X, expected_dimension) in cases
        @testset "$(typeof(S.C)) / $(length(S)) dimensions" begin

            chart = Copulas._sklar_joint_chart(S)
            z = Copulas._sklar_joint_coordinates(chart)

            @test Copulas._sklar_joint_optimizer_kind(chart) === :lbfgs
            @test length(z) == expected_dimension
            @test length(chart.ranges) == length(S.m) + 1
            @test all(isfinite, z)

            recovered = Copulas._sklar_joint_distribution(chart, z)

            @test recovered isa SklarDist
            @test params(recovered.C) == params(S.C)

            for i in eachindex(S.m)
                @test all(isapprox(a, b; atol=1e-12, rtol=1e-12) for (a, b) in zip(params(recovered.m[i]), params(S.m[i]),))
            end

            objective(v) = -sum(logpdf(Copulas._sklar_joint_distribution(chart, v), X,))
            # The joint objective must be evaluable in ordinary real arithmetic.
            @test isfinite(objective(z))

            # Independent finite-difference gradient for every supported chart.
            h = 1e-5

            numerical_gradient = map(eachindex(z)) do j
                shift = zeros(eltype(z), length(z))
                shift[j] = h

                return (objective(z + shift) -objective(z - shift)) / (2h)
            end

            @test length(numerical_gradient) == expected_dimension
            @test all(isfinite, numerical_gradient)

            # Gamma CDF differentiation with respect to its shape parameter
            # is not currently supported by ForwardDiff/SpecialFunctions.
            # Its numerical gradient is tested above; ForwardDiff is tested
            # where the complete joint CDF path supports Dual numbers.
            if all(m -> !(m isa Gamma), S.m)
                gradient = ForwardDiff.gradient(objective, z)

                @test all(isfinite, gradient)
                @test all(isapprox(a, b; rtol=2e-4, atol=2e-5) for (a, b) in zip(gradient, numerical_gradient))
            end

        end
    end
end

@testset "Generic JMLE differentiation selection" begin
    rng = MersenneTwister(2031)
    cases = (
        (Normal(), Copulas.ADTypes.AutoForwardDiff),
        (Exponential(), Copulas.ADTypes.AutoForwardDiff),
        (Weibull(1.8, 1.2), Copulas.ADTypes.AutoForwardDiff),
        (LogNormal(0.2, 0.9), Copulas.ADTypes.AutoForwardDiff),
        (Gamma(2.5, 1.1), Copulas.ADTypes.AutoFiniteDiff),
        (Beta(2.3, 3.7), Copulas.ADTypes.AutoFiniteDiff),
    )

    for (margin, autodiff_type) in cases
        S = SklarDist(ClaytonCopula{2}(0.8), (Normal(0.0, 1.5), margin))
        X = rand(rng, S, 24)
        chart = Copulas._sklar_joint_chart(S)
        z = Copulas._sklar_joint_coordinates(chart)
        objective(v) = Copulas._sklar_joint_objective(chart, X, nothing, v)
        autodiff = Copulas._sklar_joint_lbfgs_autodiff(objective, z)

        @test autodiff isa autodiff_type
    end

    @test Copulas._sklar_joint_lbfgs_autodiff(_ -> Inf, [0.0]) isa Copulas.ADTypes.AutoFiniteDiff
    @test_throws ErrorException Copulas._sklar_joint_lbfgs_autodiff(_ -> error("test error"), [0.0])
end

@testset "JMLE numerical recovery" begin
    rng = MersenneTwister(2032)

    S3 = SklarDist(GaussianCopula([
        1.0 0.30 -0.20
        0.30 1.0 0.45
        -0.20 0.45 1.0]), (Normal(), Normal(), Normal()))
    X3 = rand(rng, S3, 45)
    chart3 = Copulas._sklar_joint_chart(S3)
    zbad = [79.50806956416622, -8.299203551788883, 113.12433239604641,
            66.87414464081553, -123.57518091806037, 53.16701546200909,
            -116.440604901349, 73.68877658770283, -40.27763925745332]
    @test Copulas._sklar_joint_objective(chart3, X3, nothing, zbad) == Inf

    S5 = SklarDist(GaussianCopula([
        1.0 0.20 -0.15 0.10 0.30
        0.20 1.0 0.40 -0.25 0.05
        -0.15 0.40 1.0 0.35 -0.10
        0.10 -0.25 0.35 1.0 0.20
        0.30 0.05 -0.10 0.20 1.0]), ntuple(_ -> Normal(), 5))
    X5 = rand(rng, S5, 45)
    chart5 = Copulas._sklar_joint_chart(S5)
    z5 = Copulas._sklar_joint_coordinates(chart5)
    @test Copulas._sklar_joint_lbfgs_autodiff(
        z -> Copulas._sklar_joint_objective(chart5, X5, nothing, z), z5;
        chart=chart5, X=X5, weights=nothing) isa Copulas.ADTypes.AutoFiniteDiff

    for (S, X, expected) in ((S3, X3, 9), (S5, X5, 20))
        M = fit(CopulaModel, S, X; sklar_method=:jmle)
        ll = loglikelihood(M)
        @test StatsBase.dof(M) == expected
        @test length(StatsBase.coef(M)) == expected
        @test isapprox(StatsBase.aic(M), 2 * expected - 2 * ll)
        @test isapprox(StatsBase.bic(M), expected * log(size(X, 2)) - 2 * ll)
        @test isfinite(ll)
    end
end

@testset "Constructor-oracle marginal geometry" begin
    margins = (
        Cauchy(0.5, 1.3),
        Laplace(-0.3, 1.7),
        TDist(7.0),
        Poisson(2.5),
        Binomial(10, 0.4),
        Uniform(-2.0, 3.0),
    )

    for margin in margins
        coordinates = Copulas._sklar_constructor_oracle_coordinates(margin)
        reconstructed = Copulas._sklar_constructor_oracle_from_coordinates(margin, coordinates)

        @test typeof(reconstructed) === typeof(margin)
        @test params(reconstructed) == params(margin)
        @test params(typeof(margin)(params(margin)...)) == params(margin)
    end

    @test length(Copulas._sklar_constructor_oracle_coordinates(Binomial(10, 0.4))) == 1
    fixed = Copulas._sklar_constructor_oracle_from_coordinates(Binomial(10, 0.4), [0.7])
    @test params(fixed)[1] == 10
    @test params(fixed)[2] == 0.7

    err = try
        Copulas._sklar_constructor_oracle_coordinates(DiscreteUniform(1, 10))
    catch exception
        exception
    end
    @test err isa ArgumentError
    @test occursin("no floating-point parameters", sprint(showerror, err))
    @test_throws DimensionMismatch Copulas._sklar_constructor_oracle_from_coordinates(Poisson(2.5), Float64[])
    @test_throws DomainError Copulas._sklar_constructor_oracle_from_coordinates(Uniform(0.0, 1.0), [2.0, 1.0])

    fixed_margin = Kolmogorov()
    fixed_chart = Copulas._sklar_joint_chart(
        SklarDist(IndependentCopula{2}(), (fixed_margin, Normal())))
    @test fixed_chart.margin_modes == (:fixed, :paramorph)
    @test Copulas._sklar_joint_optimizer_kind(fixed_chart) === :lbfgs
    @test Copulas._sklar_joint_distribution(fixed_chart, Copulas._sklar_joint_coordinates(fixed_chart)).m[1] === fixed_margin
end

@testset "Constructor-oracle JMLE fallback" begin
    rng = MersenneTwister(2028)

    @testset "Clayton with Cauchy and Laplace" begin
        T = SklarDist{ClaytonCopula,Tuple{Cauchy,Laplace}}
        S = SklarDist(ClaytonCopula{2}(1.2), (Cauchy(0.5, 0.9), Laplace(-0.4, 1.2)))
        X = rand(rng, S, 100)
        M = fit(CopulaModel, T, X; sklar_method=:jmle)

        chart = Copulas._sklar_joint_chart(fitted_distribution(M))
        @test chart.margin_modes == (:constructor_oracle, :constructor_oracle)
        @test Copulas._sklar_joint_optimizer_kind(chart) === :nelder_mead
        @test fitted_distribution(M).m[1] isa Cauchy
        @test fitted_distribution(M).m[2] isa Laplace
        @test Copulas.fitting_method(M) === :jmle
        @test isfinite(loglikelihood(M))
    end

    @testset "Clayton with Normal and fixed Binomial n" begin
        S = SklarDist(ClaytonCopula{2}(1.1), (Normal(0.5, 1.0), Binomial(10, 0.65)))
        S0 = SklarDist(ClaytonCopula{2}(0.8), (Normal(0.0, 1.5), Binomial(10, 0.4)))
        X = rand(rng, S, 100)
        fitted = Copulas._fit_sklar_joint_mle(S0, X)

        chart = Copulas._sklar_joint_chart(fitted)
        @test chart.margin_modes == (:paramorph, :constructor_oracle)
        @test Copulas._sklar_joint_optimizer_kind(chart) === :nelder_mead
        @test params(fitted.m[2])[1] == 10
        @test isfinite(loglikelihood(fitted, X))
    end

    @testset "Weighted fallback preserves initial likelihood" begin
        S = SklarDist(ClaytonCopula{2}(1.2), (Cauchy(0.5, 0.9), Laplace(-0.4, 1.2)))
        S0 = SklarDist(ClaytonCopula{2}(0.8), (Cauchy(0.0, 2.0), Laplace(0.0, 2.0)))
        X = rand(rng, S, 90)
        weights = repeat([0.0, 1.0, 2.0], 30)
        initial = -Copulas._weighted_loglikelihood(S0, X, weights)
        fitted = Copulas._fit_sklar_joint_mle(S0, X; weights)
        final = -Copulas._weighted_loglikelihood(fitted, X, weights)

        @test Copulas._sklar_joint_optimizer_kind(Copulas._sklar_joint_chart(fitted)) === :nelder_mead
        @test isfinite(final)
        @test final <= initial + 1e-7
    end
end

@testset "JMLE from an initial SklarDist" begin
    rng = MersenneTwister(2029)
    S = SklarDist(ClaytonCopula{2}(1.1), (Normal(0.5, 1.0), Binomial(10, 0.65)))
    S0 = SklarDist(ClaytonCopula{2}(0.8), (Normal(0.0, 1.5), Binomial(10, 0.4)))
    X = rand(rng, S, 90)
    weights = repeat([0.0, 1.0, 2.0], 30)
    normalized_weights = Copulas._fit_weights(weights, size(X, 2))
    initial = -Copulas._weighted_loglikelihood(S0, X, normalized_weights)

    fitted = fit(S0, X; sklar_method=:jmle, weights)
    @test fitted isa SklarDist
    @test params(fitted.m[2])[1] == 10
    @test -Copulas._weighted_loglikelihood(fitted, X, normalized_weights) <= initial + 1e-7

    model = fit(CopulaModel, S0, X; sklar_method=:jmle, weights)
    @test fitted_distribution(model) isa SklarDist
    @test model.recipe.target === S0
    @test params(fitted_distribution(model).m[2])[1] == 10
    @test isapprox(loglikelihood(model), Copulas._weighted_loglikelihood(fitted_distribution(model), X, normalized_weights); atol=1e-10)

    refit = Copulas._refit(model, X; replay_input=true)
    @test refit isa CopulaModel
    @test params(fitted_distribution(refit).m[2])[1] == 10
    @test Copulas.fitting_method(refit) === :jmle

    @test_throws ArgumentError fit(S0, X; sklar_method=:ifm)
    @test_throws DimensionMismatch fit(S0, rand(rng, 3, 20); sklar_method=:jmle)
end

@testset "S0 residuals and constructor-oracle dof" begin
    rng = MersenneTwister(2030)

    S0_continuous = SklarDist(IndependentCopula{2}(), (Normal(0.0, 1.5), Normal(0.0, 1.5)))
    X_continuous = rand(rng, S0_continuous, 60)
    M_continuous = fit(CopulaModel, S0_continuous, X_continuous; sklar_method=:jmle)
    residuals = StatsBase.residuals(M_continuous)

    @test size(residuals) == size(X_continuous)
    @test all(isfinite, residuals)

    S0_erlang = SklarDist(IndependentCopula{2}(), (Erlang(3, 1.0), Normal()))
    X_erlang = rand(rng, S0_erlang, 80)
    M_erlang = fit(CopulaModel, S0_erlang, X_erlang; sklar_method=:jmle)
    ll = loglikelihood(M_erlang)

    @test StatsBase.dof(M_erlang) == 3
    @test length(StatsBase.coef(M_erlang)) == 4
    @test isapprox(StatsBase.aic(M_erlang), 2 * 3 - 2 * ll)
    @test isapprox(StatsBase.bic(M_erlang), 3 * log(80) - 2 * ll)

    S0_fixed = SklarDist(IndependentCopula{2}(), (Kolmogorov(), Normal()))
    X_fixed = rand(rng, S0_fixed, 60)
    M_fixed = fit(CopulaModel, S0_fixed, X_fixed; sklar_method=:jmle)

    @test params(fitted_distribution(M_fixed).m[1]) == ()
    @test StatsBase.dof(M_fixed) == 2
    @test isfinite(loglikelihood(M_fixed))
end

@testset "Sklar joint maximum likelihood" begin

    @testset "Clayton with continuous margins" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.5), (Normal(), Exponential()))
        X = rand(rng, S, 150)

        M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
        M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)
        fitted = fitted_distribution(M_jmle)

        @test fitted isa SklarDist
        @test Copulas.fitting_method(M_jmle) === :jmle
        @test StatsBase.dof(M_jmle) == 4
        @test StatsBase.nobs(M_jmle) == 150
        @test isfinite(loglikelihood(M_jmle))
        @test loglikelihood(M_jmle) >= loglikelihood(M_ifm) - 1e-6
        @test all(isfinite, StatsBase.coef(M_jmle))

        direct = fit(T, X; sklar_method=:jmle)
        @test direct isa SklarDist
        @test isapprox(loglikelihood(direct, X), loglikelihood(M_jmle); atol=1e-6)
    end

    @testset "Independent copula oracle" begin
        rng = MersenneTwister(2026)
        T = SklarDist{IndependentCopula,Tuple{Normal,Exponential}}
        S = SklarDist(IndependentCopula{2}(), (Normal(1.0, 1.5), Exponential(2.0)))
        X = rand(rng, S, 150)

        M = fit(CopulaModel, T, X; sklar_method=:jmle)
        m1 = fit_mle(Normal, vec(X[1, :]))
        m2 = fit_mle(Exponential, vec(X[2, :]))
        reference = sum(logpdf(m1, X[1, j]) + logpdf(m2, X[2, j]) for j in axes(X, 2))

        @test fitted_distribution(M).C isa IndependentCopula
        @test StatsBase.dof(M) == 3
        @test isapprox(loglikelihood(M), reference; atol=1e-4)
    end

    @testset "Gamma margin with numerical gradients" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential,Gamma}}
        S = SklarDist(ClaytonCopula{3}(1.2), (Normal(), Exponential(), Gamma(2.5, 1.4)))
        X = rand(rng, S, 120)

        M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
        M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)

        @test Copulas.fitting_method(M_jmle) === :jmle
        @test StatsBase.dof(M_jmle) == 6
        @test isfinite(loglikelihood(M_jmle))
        @test loglikelihood(M_jmle) >= loglikelihood(M_ifm) - 1e-6
    end

    @testset "Weights and estimator replay" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.2), (Normal(0.4, 1.1), Exponential(1.7)))
        X = rand(rng, S, 90)

        w = repeat([0.0, 1.0, 2.0], 30)
        idx = vcat((fill(j, Int(w[j])) for j in eachindex(w))...)
        Xrep = X[:, idx]

        Mw = fit(CopulaModel, T, X; sklar_method=:jmle, weights=w)
        Mr = fit(CopulaModel, T, Xrep; sklar_method=:jmle)
        replay = Copulas._refit(Mw, Xrep; replay_input=true)

        @test isapprox(loglikelihood(Mw), loglikelihood(Mr); atol=1e-3)
        @test isapprox(loglikelihood(replay), loglikelihood(Mr); atol=1e-3)
        @test Copulas.fitting_method(replay) === :jmle
    end

    @testset "Bootstrap integration" begin
        rng = MersenneTwister(2026)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        S = SklarDist(ClaytonCopula{2}(1.5), (Normal(), Exponential()))
        X = rand(rng, S, 150)

        M = fit(CopulaModel, T, X; sklar_method=:jmle)
        I = infer(M; method=:bootstrap, nresamples=4, rng=MersenneTwister(2027))
        V = StatsBase.vcov(I)

        @test size(V) == (4, 4)
        @test all(isfinite, V)
        @test isapprox(V, transpose(V); atol=1e-10)
    end

    @testset "Shared marginal parameter geometry" begin
        families = (Normal(), Exponential(), Gamma(), Beta(2.3, 3.7),
                    Weibull(1.8, 2.1), LogNormal(0.4, 1.2))

        for margin in families
            z = Copulas._sklar_margin_coordinates(margin)
            reconstructed = Copulas._sklar_margin_from_coordinates(margin, z)
            @test all(isapprox.(params(reconstructed), params(margin); rtol=1e-10))
        end
    end

    @testset "Extended marginal families" begin
        rng = MersenneTwister(2026)
        cases = ((Beta(2.3, 3.7), Beta), (Weibull(1.8, 2.1), Weibull),
                (LogNormal(0.4, 1.2), LogNormal))

        for (margin, family) in cases
            T = SklarDist{ClaytonCopula,Tuple{family,Normal}}
            S = SklarDist(ClaytonCopula{2}(1.2), (margin, Normal(0.5, 1.3)))
            X = rand(rng, S, 150)

            M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
            M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)

            @test Copulas._sklar_joint_optimizer_kind(Copulas._sklar_joint_chart(fitted_distribution(M_jmle))) === :lbfgs
            @test Copulas.fitting_method(M_jmle) === :jmle
            @test StatsBase.dof(M_jmle) == 5
            @test isfinite(loglikelihood(M_jmle))
            @test loglikelihood(M_jmle) >= loglikelihood(M_ifm) - 1e-6
        end
    end
end
