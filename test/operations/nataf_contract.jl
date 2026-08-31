# Nataf-operation contract: matrix structure, attainable ranges, and documented
# validation errors for `Nataf`.
@testset "Nataf structure and validation" begin
    R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3)), [1.0 0.0; 0.0 1.0])
    @test R₀ == [1.0 0.0; 0.0 1.0]

    R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3), Beta(2, 5)),
        [1.0 0.5 0.0; 0.5 1.0 -0.3; 0.0 -0.3 1.0])
    @test LinearAlgebra.issymmetric(R₀)
    @test all(LinearAlgebra.diag(R₀) .== 1)
    @test R₀[1, 3] == 0

    @test_throws ArgumentError Nataf((Uniform(), Normal()), 0.99)

    margins = (LogNormal(0, 0.8), Gamma(2, 3))
    @test_throws ArgumentError Nataf((Pareto(1.0), Normal()), 0.5)
    @test_throws ArgumentError Nataf(
        (LogNormal(0, 2), LogNormal(0, 2)), -0.5)
    @test_throws ArgumentError Nataf(margins, [1.0 0.5; 0.4 1.0])
    @test_throws ArgumentError Nataf(margins, [0.9 0.5; 0.5 1.0])
    @test_throws ArgumentError Nataf(margins,
        [1.0 0.5 0.1; 0.5 1.0 0.1; 0.1 0.1 1.0])
    @test_throws ArgumentError Nataf(
        (Pareto(1.0), Normal()), [1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError Nataf((Normal(0, 0), Normal()), 0.5)
    @test_throws ArgumentError Nataf(
        (LogNormal(0, 0), LogNormal(0, 1)), 0.3)
    @test_throws ArgumentError Nataf(Normal(), [1.0 0.5; 0.5 1.0])
end
