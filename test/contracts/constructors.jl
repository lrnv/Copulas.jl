function test_constructor_case(case)
    @testset "$(case.name)" begin
        typed = case.inferred ? (@inferred case.typed()) : case.typed()
        dynamic = case.dynamic()
        @test typed == dynamic
        @test typeof(typed) === typeof(dynamic)
        @test params(typed) == params(dynamic)
    end
end

@testset "public constructors" begin
    for case in CONSTRUCTOR_CASES
        test_constructor_case(case)
    end
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
