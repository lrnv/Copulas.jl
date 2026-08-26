@testset "public constructors" begin
    for case in CONSTRUCTOR_CASES
        @testset "$(case.name)" begin
            typed = @inferred case.typed()
            dynamic = case.dynamic()
            @test typed == dynamic
            @test typeof(typed) === typeof(dynamic)
        end
    end
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
