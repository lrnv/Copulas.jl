@testset "Public docstring coverage" begin
    for name in _PUBLIC_SYMBOLS
        @test isdefined(Copulas, name)
        @test Base.Docs.hasdoc(Copulas, name)
    end

    # Public behavior also includes methods adopted from ecosystem interfaces.
    adopted = (
        Distributions.fit,
        Distributions.params,
        StatsBase.nobs,
        StatsBase.isfitted,
        StatsBase.deviance,
        StatsBase.dof,
        StatsBase.coef,
        StatsBase.coefnames,
        StatsBase.vcov,
        StatsBase.stderror,
        StatsBase.confint,
        StatsBase.aic,
        StatsBase.bic,
        StatsBase.predict,
        StatsBase.residuals,
    )
    @test all(!isnothing(Base.Docs.doc(f)) for f in adopted)
end

@testset "Developer-guide protocol docstrings" begin
    guide_bindings = (
        :CopulaMeasureStyle,
        :LimitKind,
        :copula_measure_style,
        :_cdf,
        :distortion,
        :_partial_cdf,
        :conditional_copula,
        :_mixed_partial,
        :Distortion,
        :DistortedDist,
        :ConditionalCopula,
        :SubsetCopula,
        :_available_fitting_methods,
        :_fit,
        :_example,
        :_unbound_params,
        :_rebound_params,
        :CopulaHypothesis,
        :_run_copula_test,
        :_teststatistic,
        :_calibrate,
        :_test_method,
        :_bootstrap_hypothesis,
        :testname,
        :nullhypothesis,
        :BivariatePickandsTail,
        :BivEVDistortion,
        :_is_valid_in_dim,
        :dA,
        :d²A,
        :ellpartial,
        :_ellpartial_signlog,
        :_discrete_spectral_rand!,
        :limit_kind,
        :EllipticalCopula,
        :U,
        :N,
        :make_cor!,
        :AbstractUnivariateGenerator,
        :ϕ⁻¹,
        :ϕ⁽¹⁾,
        :ϕ⁻¹⁽¹⁾,
        :ϕ⁽ᵏ⁾,
        :ϕ⁽ᵏ⁾⁻¹,
    )

    for name in guide_bindings
        @test isdefined(Copulas, name)
        @test Base.Docs.hasdoc(Copulas, name)
    end
end
