@testset "Automatic copula-family selection" begin
    U = rand(rng, ClaytonCopula(2, 6.0), 300)

    candidates = (
        IndependentCopula,
        ClaytonCopula,
    )

    @testset "BIC selection" begin
        M = fit(
            CopulaModel,
            Copula,
            U;
            candidates=candidates,
            criterion=:bic,
            vcov=false,
        )

        @test M isa CopulaModel
        @test M.method_details.selection === true
        @test M.method_details.criterion === :bic
        @test M.method_details.candidates == candidates
        @test M.method_details.selected_family === ClaytonCopula
        @test M.method_details.selected_index in eachindex(candidates)
        @test isfinite(M.method_details.selected_score)

        table = selectiontable(M)

        @test table isa AbstractVector
        @test length(table) == length(candidates)
        @test table.criterion === :bic
        @test table.selected_family === ClaytonCopula

        @test [row.candidate for row in table] == collect(candidates)
        @test all(row -> row.status === :ok, table)
        @test all(row -> isfinite(row.loglikelihood), table)
        @test all(row -> isfinite(row.bic), table)

        selected_row = only(filter(
            row -> row.candidate === ClaytonCopula,
            table,
        ))

        @test selected_row.bic == minimum(row.bic for row in table)
        @test M.method_details.selected_score == selected_row.bic
    end

    @testset "Selection table display" begin
        M = fit(
            CopulaModel,
            Copula,
            U;
            candidates=candidates,
            vcov=false,
        )

        table = selectiontable(M)

        io = IOBuffer()
        show(io, MIME("text/plain"), table)
        printed = String(take!(io))

        @test occursin("Copula model selection", printed)
        @test occursin("BIC", printed)
        @test occursin("IndependentCopula", printed)
        @test occursin("ClaytonCopula", printed)
        @test occursin("selected model", printed)

        io = IOBuffer()
        show(io, MIME("text/plain"), M)
        printed = String(take!(io))

        @test occursin("Model selection", printed)
        @test occursin("Criterion:", printed)
        @test occursin("Selected family:", printed)
    end

    @testset "Information criteria" begin
        for criterion in (:aic, :aicc, :hqc)
            M = fit(
                CopulaModel,
                Copula,
                U;
                candidates=candidates,
                criterion=criterion,
                vcov=false,
            )

            table = selectiontable(M)

            @test M.method_details.criterion === criterion
            @test table.criterion === criterion

            eligible = filter(row -> row.status === :ok, table)
            values = getproperty.(eligible, criterion)

            @test M.method_details.selected_score == minimum(values)
        end
    end

    @testset "API errors" begin
        fitted = fit(
            CopulaModel,
            ClaytonCopula,
            U;
            vcov=false,
        )

        @test_throws ArgumentError selectiontable(fitted)

        @test_throws ArgumentError fit(
            CopulaModel,
            Copula,
            U;
            candidates=candidates,
            criterion=:invalid,
            vcov=false,
        )

        @test_throws ArgumentError fit(
            CopulaModel,
            Copula,
            U;
            candidates=candidates,
            on_error=:invalid,
            vcov=false,
        )

        @test_throws ArgumentError fit(
            CopulaModel,
            Copula,
            U;
            candidates=(Normal,),
            vcov=false,
        )

        @test_throws ArgumentError fit(
            CopulaModel,
            Copula,
            U;
            candidates=(Copula,),
            vcov=false,
        )
    end
end
