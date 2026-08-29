# Specialization-equivalence obligation: a flat nested declaration reduces to
# the native Archimedean representation without changing its density route.
@testset "flat nested declarations dispatch to native copulas" begin
    cases = (
        (NestedArchimedeanCopula(Copulas.ClaytonGenerator(2.0);
             leaves=[1, 2, 3]), ClaytonCopula{3}(2.0), [0.31, 0.53, 0.79]),
        (NestedArchimedeanCopula(Copulas.GumbelGenerator(2.5);
             leaves=[1, 2, 3, 4]), GumbelCopula{4}(2.5),
             [0.27, 0.43, 0.61, 0.82]),
    )
    for (reduced, native, u) in cases
        @test typeof(reduced) == typeof(native)
        @test !(reduced isa NestedArchimedeanCopula)
        @test logpdf(reduced, u) === logpdf(native, u)
    end
end
