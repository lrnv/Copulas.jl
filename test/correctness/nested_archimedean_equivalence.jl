# A flat nested declaration retains the nested family type while composing
# to the same distribution as the native Archimedean copula.
@testset "flat nested declarations retain type and match native copulas" begin
    cases = (
        (NestedArchimedeanCopula(Copulas.ClaytonGenerator(2.0);
             leaves=[1, 2, 3]), ClaytonCopula{3}(2.0), [0.31, 0.53, 0.79]),
        (NestedArchimedeanCopula(Copulas.GumbelGenerator(2.5);
             leaves=[1, 2, 3, 4]), GumbelCopula{4}(2.5),
             [0.27, 0.43, 0.61, 0.82]),
    )
    for (nested, native, u) in cases
        @test nested isa NestedArchimedeanCopula
        @test length(nested) == length(native)
        @test cdf(nested, u) ≈ cdf(native, u) atol=2e-14 rtol=2e-14
        @test logpdf(nested, u) ≈ logpdf(native, u) atol=2e-14 rtol=2e-14
    end
end
