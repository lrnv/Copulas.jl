# Public selection contract and selection-specific decisions. Estimator accuracy
# is tested in fitting.jl; reuse fits here rather than duplicating those oracles.
struct SelectionProbe{mode} <: Copula{2} end
const SELECTION_PROBE_CALLS = Ref(0)
function Distributions.fit(::Type{CopulaModel}, ::Type{SelectionProbe{mode}}, U; kwargs...) where {mode}
    SELECTION_PROBE_CALLS[] += 1
    mode === :interrupt && throw(InterruptException())
    mode === :bad_score && return CopulaModel(SelectionProbe{:bad_score}(), size(U, 2), 0.0, :probe)
    return CopulaModel(IndependentCopula{2}(), size(U, 2),
        mode === :nonfinite ? NaN : 0.0, :probe;
        converged=mode !== :not_converged, iterations=7, method_details=(; U))
end
StatsBase.coef(::CopulaModel{SelectionProbe{:bad_score}}) = throw(ArgumentError("unavailable parameter count"))

@testset "Automatic copula-family selection" begin
    U = rand(StableRNG(436), ClaytonCopula{2}(6.0), 80)
    candidates = (IndependentCopula, ClaytonCopula)
    M = fit(CopulaModel, Copula, U; candidates, vcov=false, derived_measures=false)
    table = selectiontable(M)
    @test M isa CopulaModel
    @test table isa Vector
    @test getproperty.(table, :candidate) == collect(candidates)
    @test all(row -> row.status === :ok, table)
    @test table[M.method_details.selected_index].bic == minimum(row.bic for row in table)
    @test loglikelihood(M) == table[M.method_details.selected_index].loglikelihood
    @test M.result isa ClaytonCopula
    @test occursin("Model selection", sprint(show, M))
    displayed = sprint(show, M)
    reverse!(table)
    @test getproperty.(selectiontable(M), :candidate) == collect(candidates)
    @test sprint(show, M) == displayed
    @test_throws ArgumentError GOFCopulaTest(M; N=1)
    @test_throws ArgumentError GOFCopulaTest(M, U; N=1)

    @testset "Information criterion $criterion" for criterion in (:aic, :aicc, :hqc)
        selected = fit(CopulaModel, Copula, U; candidates, criterion,
            vcov=false, derived_measures=false)
        rows = selectiontable(selected)
        @test getproperty(rows[selected.method_details.selected_index], criterion) ==
            minimum(getproperty(row, criterion) for row in rows)
    end

    @testset "Validation and failed candidates" begin
        score_failure = fit(CopulaModel, Copula, U;
            candidates=(SelectionProbe{:bad_score}, IndependentCopula), vcov=false)
        @test first(selectiontable(score_failure)).status === :failed
        @test occursin("unavailable parameter count", first(selectiontable(score_failure)).error)
        @test_throws ArgumentError fit(CopulaModel, Copula, U;
            candidates=(SelectionProbe{:bad_score},), on_error=:throw)
        SELECTION_PROBE_CALLS[] = 0
        fit(CopulaModel, Copula, U; candidates=(SelectionProbe{:ok},), vcov=false)
        @test SELECTION_PROBE_CALLS[] == 1
        @test_throws InterruptException fit(CopulaModel, Copula, U;
            candidates=(SelectionProbe{:interrupt},), vcov=false)
        for mode in (:nonfinite, :not_converged)
            probe = fit(CopulaModel, Copula, U;
                candidates=(SelectionProbe{mode}, IndependentCopula), vcov=false)
            @test selectiontable(probe)[1].status === mode
            @test probe.method_details.selected_index == 2
        end
        relaxed = fit(CopulaModel, Copula, U;
            candidates=(SelectionProbe{:not_converged},), require_convergence=false, vcov=false)
        @test !relaxed.converged
        @test relaxed.iterations == 7
        @test only(selectiontable(relaxed)).status === :ok
        @test_throws ArgumentError selectiontable(
            fit(CopulaModel, IndependentCopula, U; vcov=false))
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates=())
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates=(Normal,))
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates=(Copula,))
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates, criterion=:invalid)
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates, on_error=:invalid)
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates, vcov_method=:invalid)
        # Abstract families cannot be fitted without a model specification.
        invalid = (Copulas.SubsetCopula, IndependentCopula)
        skipped = fit(CopulaModel, Copula, U; candidates=invalid, vcov=false)
        @test selectiontable(skipped)[1].status === :failed
        @test selectiontable(skipped)[2].status === :ok
        @test_throws ArgumentError fit(CopulaModel, Copula, U;
            candidates=invalid, on_error=:throw)
        @test_throws ArgumentError fit(CopulaModel, Copula, U; candidates=(Copulas.SubsetCopula,))
        @test fit(Copula, U; candidates=(IndependentCopula,)) isa IndependentCopula
        # An undefined small-sample AICc excludes the fit.
        @test_throws ArgumentError fit(CopulaModel, Copula, U[:, 1:1];
            candidates=(IndependentCopula,), criterion=:aicc)
    end
end
