function test_constructor_case(case)
    @testset "$(case.name)" begin
        typed = @inferred case.typed()
        dynamic = case.dynamic()
        @test typed == dynamic
        @test typeof(typed) === typeof(dynamic)
    end
end

@testset "public constructors" begin
    for case in CONSTRUCTOR_CASES
        test_constructor_case(case)
    end
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
