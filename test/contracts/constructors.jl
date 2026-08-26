function test_constructor_case(case)
    typed = Ref{Any}()
    @testset "$(case.name)" begin
        typed[] = case.inferred ? (@inferred case.typed()) : case.typed()
        typed_value = typed[]
        dynamic = case.dynamic()
        @test typed_value == dynamic
        @test typeof(typed_value) === typeof(dynamic)
        @test params(typed_value) == params(dynamic)
        if case.reconstruct
            reconstructed = typeof(typed_value)(values(params(typed_value))...)
            @test typeof(reconstructed) === typeof(typed_value)
            @test params(reconstructed) == params(typed_value)
        end
    end
    return typed[]
end

@testset "public constructors" begin
    constructed = map(test_constructor_case, CONSTRUCTOR_CASES)
    public_families = [getfield(Copulas, symbol) for symbol in PUBLIC_SYMBOLS
        if Base.isexported(Copulas, symbol) &&
           getfield(Copulas, symbol) isa Type &&
           getfield(Copulas, symbol) <: Copulas.Copula]
    @test length(CONSTRUCTOR_CASES) == length(public_families)
    @test all(F -> any(C -> C isa F, constructed), public_families)
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
