@testset "Liebscher and Khoudraji constructor validation" begin
    C2a = ClaytonCopula{2}(1.5)
    C2b = GumbelCopula{2}(1.4)
    C3 = ClaytonCopula{3}(1.5)

    @test_throws ArgumentError LiebscherCopula(2, (), zeros(0, 2))
    @test_throws DimensionMismatch LiebscherCopula(2, (C2a, C3), [0.5 0.5; 0.5 0.5])
    @test_throws DimensionMismatch LiebscherCopula(2, (C2a, C2b), ones(3, 2) ./ 3)
    @test_throws DomainError LiebscherCopula(2, (C2a, C2b), [-0.1 0.5; 1.1 0.5])
    @test_throws ArgumentError LiebscherCopula(2, (C2a, C2b), [0.4 0.5; 0.5 0.5])
    @test_throws ArgumentError LiebscherCopula(2, (C2a, C2b), [Inf 0.5; 0.0 0.5])

    @test_throws DimensionMismatch KhoudrajiCopula(2, C3, [0.3, 0.7])
    @test_throws DimensionMismatch KhoudrajiCopula(2, (C2a, C3), [0.3, 0.7])
    @test_throws DimensionMismatch KhoudrajiCopula(2, C2a, [0.3])
    @test_throws DimensionMismatch KhoudrajiCopula(2, (C2a, C2b), [0.3])
    @test_throws DomainError KhoudrajiCopula(2, C2a, [-0.1, 0.7])
    @test_throws DomainError KhoudrajiCopula(2, (C2a, C2b), [0.3, 1.1])
    @test_throws ArgumentError KhoudrajiCopula(2, C2a, [NaN, 0.7])
    @test_throws ArgumentError KhoudrajiCopula(2, (C2a, C2b), [Inf, 0.7])

    @test_throws MethodError KhoudrajiCopula(C2a, [0.3, 0.7])
    @test_throws MethodError KhoudrajiCopula(C2a, C2b, [0.3, 0.7])
    @test_throws MethodError KhoudrajiCopula(2, (C2a,), [0.3, 0.7])
end
