# Family-regression layer: nested-specific boundary propagation that is not a
# constructor contract or a generic copula identity.
@testset "nested Archimedean boundary regressions" begin
    C = NestedArchimedeanCopula(Copulas.ClaytonGenerator(2.0);
        children=[ClaytonCopula{2}(5.0), ClaytonCopula{2}(6.0)])
    u = [0.3, 0.4, 0.6, 0.7]
    @test cdf(C, [u[1], u[2], 1.0, 1.0]) ≈
          cdf(ClaytonCopula{2}(5.0), u[1:2])
    for point in ([0, 1, 1, 1], [u[1], 1.0, u[3], u[4]],
                  [u[1], -0.1, u[3], u[4]], [u[1], NaN, u[3], u[4]])
        @test logpdf(C, point) == -Inf
    end
end
