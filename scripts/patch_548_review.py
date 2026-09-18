from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one match in {path}, found {text.count(old)}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "src/Fitting.jl",
    '''# Weights recorded by the estimator that produced a model, or `nothing`.
_model_weights(M::CopulaModel) =
    M.recipe isa _CopulaFitSpec ? get(M.recipe.kwargs, :weights, nothing) : nothing
''',
    '''# Non-trivial weights recorded by the estimator that produced a model, or
# `nothing`. A positive constant vector is statistically identical to the
# unweighted sample after normalization, so downstream behavior (including
# inference and GOF availability) must not depend on whether the user supplied
# `weights=ones(n)` or any other constant positive scale.
function _model_weights(M::CopulaModel)
    M.recipe isa _CopulaFitSpec || return nothing
    weights = get(M.recipe.kwargs, :weights, nothing)
    weights === nothing && return nothing
    return all(==(first(weights)), weights) ? nothing : weights
end
''',
)

replace_once(
    "src/Fitting.jl",
    '''function _is_missing_weighted_fit(err::MethodError)
    f = err.f === Core.kwcall ? err.args[2] : err.f
    return f in (Distributions.fit, Distributions.fit_mle, Distributions.suffstats)
end
_is_missing_weighted_fit(err::ErrorException) =
    startswith(err.msg, "suffstats is not implemented")
_is_missing_weighted_fit(::Exception) = false
''',
    '''# Only Distributions.jl's explicit `suffstats` fallback means that the
# weighted capability is absent. A `MethodError` can originate inside an
# existing weighted fitting implementation and must propagate unchanged.
_is_missing_weighted_fit(err::ErrorException) =
    startswith(err.msg, "suffstats is not implemented")
_is_missing_weighted_fit(::Exception) = false
''',
)

# Add a regression test that an internal MethodError is not rewritten as a
# misleading "no weighted fit" ArgumentError, while the ordinary unsupported
# margin fallback remains user-friendly.
sklar_test = Path("test/operations/sklar_fitting_validation.jl")
text = sklar_test.read_text()
append = '''

struct _BrokenWeightedMargin <: ContinuousUnivariateDistribution end
_broken_weighted_fit_inner(::Int) = nothing
Distributions.fit(::Type{_BrokenWeightedMargin}, x, w) =
    _broken_weighted_fit_inner(:boom)

@testset "weighted margin errors distinguish capability from implementation failures" begin
    err = try
        Copulas._fit_margin(_BrokenWeightedMargin, [1.0, 2.0], ones(2))
    catch e
        e
    end
    @test err isa MethodError
    @test err.f === _broken_weighted_fit_inner
    @test_throws ArgumentError Copulas._fit_margin(Cauchy, [0.1, 0.2], ones(2))
end
'''
if "weighted margin errors distinguish capability from implementation failures" in text:
    raise RuntimeError("Sklar regression test already present")
sklar_test.write_text(text + append)

# Constant weights are semantically the unweighted sample, so jackknife must be
# available and give the same resampling covariance.
inference_test = Path("test/operations/inference.jl")
text = inference_test.read_text()
append = '''

@testset "constant weights are inference-equivalent to unweighted" begin
    n = 20
    U = rand(StableRNG(48_300), ClaytonCopula{2}(2.0), n)
    unweighted = fit(CopulaModel, ClaytonCopula, U)
    weighted = fit(CopulaModel, ClaytonCopula, U; weights=fill(3.0, n))

    @test Copulas._model_weights(weighted) === nothing
    J0 = infer(unweighted; method=:jackknife)
    Jw = infer(weighted; method=:jackknife)
    @test Jw.method === :jackknife
    @test vcov(Jw) == vcov(J0)
end
'''
if "constant weights are inference-equivalent to unweighted" in text:
    raise RuntimeError("jackknife regression test already present")
inference_test.write_text(text + append)

# Keep the manual wording aligned with the semantic distinction above.
manual = Path("docs/src/manual/fitting_interface.md")
text = manual.read_text()
text = text.replace(
    "`:jackknife` refuses a weighted model:",
    "`:jackknife` refuses a non-constant weighted model:",
)
manual.write_text(text)

# Remove this one-shot script from the resulting commit.
Path(__file__).unlink()
