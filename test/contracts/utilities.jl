@testset "standalone public utilities" begin
    X = [3.0 1.0 2.0 4.0; 2.0 4.0 1.0 3.0]
    U = pseudos(X)
    @test size(U) == size(X)
    @test all(x -> 0 < x < 1, U)
    @test pseudos(U) == U

    C = ClaytonCopula{2}(1.5)
    @test measure(C, zeros(2), ones(2)) == 1
    @test measure(C, [0.7, 0.2], [0.4, 0.8]) == 0
    @test 0 <= measure(C, [0.2, 0.3], [0.7, 0.8]) <= 1

    target = [1.0 0.4; 0.4 1.0]
    @test Nataf((Normal(), Normal(2, 3)), target) == target
    @test Nataf((Uniform(), Uniform()), 0.4) ≈ 2sinpi(0.4 / 6)
end
