# Public selection contract and selection-specific decisions. Estimator accuracy
# is tested in fitting.jl; reuse fits here rather than duplicating those oracles.
struct SelectionProbe{mode} <: Copulas.Copula{2} end
const SELECTION_PROBE_CALLS = Ref(0)
function Distributions.fit(::Type{CopulaModel}, ::Type{SelectionProbe{mode}}, U; kwargs...) where {mode}
    SELECTION_PROBE_CALLS[] += 1
    mode === :interrupt && throw(InterruptException())
    mode === :failed && throw(ErrorException("candidate estimator failed"))
    recipe = Copulas._CopulaFitSpec(SelectionProbe{mode}, :probe, (;))
    result = mode === :bad_score ? SelectionProbe{:bad_score}() : IndependentCopula{2}()
    return CopulaModel(result, U, mode === :nonfinite ? NaN : 0.0, recipe)
end
StatsBase.dof(::CopulaModel{SelectionProbe{:bad_score}}) = throw(ArgumentError("unavailable parameter count"))

@testset "Automatic copula-family selection" begin
    U = rand(StableRNG(436), ClaytonCopula{2}(6.0), 80)
    candidates = (IndependentCopula, ClaytonCopula)
    M = fit(CopulaModel, Copulas.Copula, U; candidates)
    table = selection_table(M)
    @test M isa Copulas.CopulaSelection
    @test selected_model(M) isa CopulaModel
    @test table isa Vector
    @test getproperty.(table, :candidate) == collect(candidates)
    @test all(row -> row.status === :ok, table)
    winner = selected_model(M)
    @test StatsBase.bic(winner) == minimum(row.bic for row in table)
    @test any(row -> row.status === :ok && row.loglikelihood ≈ loglikelihood(winner), table)
    @test fitted_distribution(winner) isa ClaytonCopula
    @test occursin("Model selection", sprint(show, M))
    displayed = sprint(show, M)
    reverse!(table)
    @test getproperty.(selection_table(M), :candidate) == collect(candidates)
    @test sprint(show, M) == displayed
    @test_throws ArgumentError GOFCopulaTest(M; N=1)
    @test_throws ArgumentError GOFCopulaTest(M, U; N=1)

    @testset "Selected fit retains the ordinary estimator" begin
        ordinary = fit(CopulaModel, ClaytonCopula, U;
            method=:mle)
        selected = fit(CopulaModel, Copulas.Copula, U; candidates=(ClaytonCopula,),
            method=:mle)
        winner = selected_model(selected)
        @test StatsBase.coef(winner) ≈ StatsBase.coef(ordinary)
        @test params(fitted_distribution(winner)) == params(fitted_distribution(ordinary))
        @test loglikelihood(winner) ≈ loglikelihood(ordinary)
        @test_throws ArgumentError infer(selected)
    end

    @testset "Information criterion $criterion" for criterion in (:aic, :aicc, :hqc)
        selected = fit(CopulaModel, Copulas.Copula, U; candidates, criterion)
        rows = selection_table(selected)
        winner = selected_model(selected)
        winner_score = criterion === :aic ? StatsBase.aic(winner) :
            criterion === :aicc ? Copulas.aicc(winner) : Copulas.hqc(winner)
        @test winner_score == minimum(getproperty(row, criterion) for row in rows)
    end

    @testset "Validation and failed candidates" begin
        score_failure = fit(CopulaModel, Copulas.Copula, U;
            candidates=(SelectionProbe{:bad_score}, IndependentCopula))
        @test first(selection_table(score_failure)).status === :failed
        @test occursin("unavailable parameter count", first(selection_table(score_failure)).error)
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U;
            candidates=(SelectionProbe{:bad_score},), on_error=:throw)
        SELECTION_PROBE_CALLS[] = 0
        fit(CopulaModel, Copulas.Copula, U; candidates=(SelectionProbe{:ok},))
        @test SELECTION_PROBE_CALLS[] == 1
        @test_throws InterruptException fit(CopulaModel, Copulas.Copula, U;
            candidates=(SelectionProbe{:interrupt},))
        for mode in (:nonfinite, :failed)
            probe = fit(CopulaModel, Copulas.Copula, U;
                candidates=(SelectionProbe{mode}, IndependentCopula))
            @test selection_table(probe)[1].status === mode
            @test fitted_distribution(selected_model(probe)) isa IndependentCopula{2}
        end
        @test_throws ArgumentError selection_table(
            fit(CopulaModel, IndependentCopula, U))
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates=())
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates=(Normal,))
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates=(Copulas.Copula,))
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates, criterion=:invalid)
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates, on_error=:invalid)
        # Abstract families cannot be fitted without a model specification.
        invalid = (Copulas.SubsetCopula, IndependentCopula)
        skipped = fit(CopulaModel, Copulas.Copula, U; candidates=invalid)
        @test selection_table(skipped)[1].status === :failed
        @test selection_table(skipped)[2].status === :ok
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U;
            candidates=invalid, on_error=:throw)
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U; candidates=(Copulas.SubsetCopula,))
        @test fit(Copulas.Copula, U; candidates=(IndependentCopula,)) isa IndependentCopula
        # An undefined small-sample AICc excludes the fit.
        @test_throws ArgumentError fit(CopulaModel, Copulas.Copula, U[:, 1:1];
            candidates=(IndependentCopula,), criterion=:aicc)
    end
end
