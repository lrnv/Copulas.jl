@testset "1.0 public exception taxonomy" begin
    C = ClaytonCopula(2, 1.2)
    badv = fill(0.5, 3)
    badm = fill(0.5, 3, 2)

    @test_throws DimensionMismatch cdf(C, badv)
    @test_throws DimensionMismatch cdf(C, badm)
    @test_throws DimensionMismatch logpdf(C, badm)

    S = SklarDist(C, (Normal(), LogNormal()))
    xbad = zeros(3)
    Xbad = zeros(3, 2)
    @test_throws DimensionMismatch cdf(S, xbad)
    @test_throws DimensionMismatch cdf(S, Xbad)
    @test_throws DimensionMismatch pdf(S, Xbad)
    @test_throws DimensionMismatch logpdf(S, Xbad)
    @test_throws DimensionMismatch rand!(rng, S, Xbad)

    # Representative specialized samplers must not change the exception type.
    @test_throws DimensionMismatch rand!(rng, MCopula(3), zeros(2, 4))
    @test_throws DimensionMismatch rand!(rng, WCopula(), zeros(3, 4))

    # Invalid public model dimensions/options are ArgumentError, not raw ErrorException.
    @test_throws ArgumentError WCopula(3)
    @test_throws ArgumentError WCopula{1}()
end
