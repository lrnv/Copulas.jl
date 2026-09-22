struct _BrokenADCopula{d,T} <: Copulas.Copula{d}
    θ::T
end

_BrokenADCopula(d::Int, θ::Float64) = _BrokenADCopula{d,Float64}(θ)
Base.eltype(C::_BrokenADCopula) = typeof(C.θ)
Distributions.params(C::_BrokenADCopula) = (C.θ,)
Distributions._logpdf(C::_BrokenADCopula, u::AbstractVector{<:Real}) = zero(eltype(u))
Copulas.Paramorph.is_paramorph_type(::Type{<:_BrokenADCopula}) = true
Copulas.Paramorph.parameter_fields(::Type{<:_BrokenADCopula}) = (:θ,)
Copulas.Paramorph.transformation_schema(::_BrokenADCopula, ::NamedTuple=NamedTuple()) =
    Copulas.Paramorph.TransformVariables.as((θ=Copulas.Paramorph.bounded_interval(
        0.0, 1.0; left_closed=false, right_closed=false,
    ),))
Copulas.Paramorph.parameter_values(C::_BrokenADCopula) = (; θ=C.θ)
Copulas.Paramorph.reconstruct_struct(C::_BrokenADCopula{d}, values::NamedTuple) where {d} =
    _BrokenADCopula(d, values.θ)
Copulas._fit_prototype(::Type{_BrokenADCopula}, ::Val{d}) where {d} =
    _BrokenADCopula(d, 0.5)
Copulas._available_fitting_methods(::Type{_BrokenADCopula}, d) = (:mle,)

struct _OverparameterizedRankCopula{d} <: Copulas.Copula{d} end
Copulas.Paramorph.is_paramorph_type(::Type{<:_OverparameterizedRankCopula}) = true
Copulas.Paramorph.transformation_schema(
    ::_OverparameterizedRankCopula, ::NamedTuple=NamedTuple(),
) = Copulas.Paramorph.TransformVariables.as((
    θ₁=Copulas.Paramorph.TransformVariables.asℝ,
    θ₂=Copulas.Paramorph.TransformVariables.asℝ,
))
Copulas._fit_prototype(
    ::Type{_OverparameterizedRankCopula}, ::Val{d},
) where {d} = _OverparameterizedRankCopula{d}()

struct _FloatOnlyNestedRecon end
(::_FloatOnlyNestedRecon)(α::AbstractVector{Float64}) = ClaytonCopula(2, exp(first(α)))

@testset "optimizer fallback does not mask implementation errors" begin
    U = [0.15 0.35 0.65 0.85; 0.25 0.55 0.45 0.75]

    # The generic Paramorph MLE starts at the unconstrained chart origin. For
    # bivariate Clayton that maps exactly to θ = 0 (independence), so both the
    # optimizer gradient and Hessian inference must be defined there.
    prototype = ClaytonCopula(2, 0.0)
    objective(α) = -Distributions.loglikelihood(
        Copulas.Paramorph.constraint(prototype, α), U)
    @test isfinite(objective([0.0]))
    @test all(isfinite, ForwardDiff.gradient(objective, [0.0]))
    @test all(isfinite, ForwardDiff.hessian(objective, [0.0]))

    # An over-parameterized generic rank fit must report the intended
    # identifiability error rather than touching the not-yet-created α₀ vector.
    rank_error = try
        Copulas._fit(_OverparameterizedRankCopula, U, Val(2), Val(:itau))
        nothing
    catch err
        err
    end
    @test rank_error isa ArgumentError
    @test occursin("2 free parameters", sprint(showerror, rank_error))
    @test occursin("1 pairwise rank constraints", sprint(showerror, rank_error))

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
