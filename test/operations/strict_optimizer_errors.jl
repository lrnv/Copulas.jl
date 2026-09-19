struct _BrokenADCopula{d,T} <: Copulas.Copula{d}
    θ::T
end

_BrokenADCopula(d::Int, θ::Float64) = _BrokenADCopula{d,Float64}(θ)
Base.eltype(C::_BrokenADCopula) = typeof(C.θ)
Distributions.params(C::_BrokenADCopula) = (C.θ,)
Distributions._logpdf(C::_BrokenADCopula, u::AbstractVector{<:Real}) = zero(eltype(u))
Copulas.Paramorph.param_space(::Type{<:_BrokenADCopula}, d) =
    Copulas.Paramorph.ProbOpen(:θ)
Copulas._available_fitting_methods(::Type{_BrokenADCopula}, d) = (:mle,)

struct _FloatOnlyNestedRecon end
(::_FloatOnlyNestedRecon)(α::AbstractVector{Float64}) = ClaytonCopula(2, exp(first(α)))

@testset "optimizer fallback does not mask implementation errors" begin
    U = [0.15 0.35 0.65 0.85; 0.25 0.55 0.45 0.75]

    # The generic Paramorph MLE starts at the unconstrained chart origin. For
    # bivariate Clayton that maps exactly to θ = 0 (independence), so both the
    # optimizer gradient and Hessian inference must be defined there.
    p = Copulas.Paramorph.param_space(ClaytonCopula, 2)
    objective(α) = -Distributions.loglikelihood(
        Copulas._parameter_space_copula(ClaytonCopula, 2, p, α), U)
    @test isfinite(objective([0.0]))
    @test all(isfinite, ForwardDiff.gradient(objective, [0.0]))
    @test all(isfinite, ForwardDiff.hessian(objective, [0.0]))

    # This model reconstructs correctly for ordinary Float64 optimizer points
    # but deliberately has no constructor accepting ForwardDiff.Dual. Before
    # #518 the generic fitter swallowed that MethodError and silently retried
    # with Nelder--Mead, making the fit appear to succeed.
    @test_throws MethodError fit(_BrokenADCopula, U; method=:mle)

    # The nested optimizer used the same exception-driven retry policy.
    @test_throws MethodError Copulas._fit_nested(_FloatOnlyNestedRecon(), [0.1], U)

    # Real package fitters still use their intended analytical MLE routes.
    @test fit(GaussianCopula, U; method=:mle) isa GaussianCopula
    @test fit(ClaytonCopula, U; method=:mle) isa ClaytonCopula
end
