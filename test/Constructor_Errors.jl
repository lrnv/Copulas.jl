@testitem "Constructors errors on wrong inputs" begin
    using LinearAlgebra
    @test_throws LinearAlgebra.PosDefException GaussianCopula([1 2.0; 2 1])
    @test_throws LinearAlgebra.PosDefException TCopula(10,[1 2.0; 2 1])
    @test_throws ArgumentError PlackettCopula(-0.5)
    @test_throws ArgumentError FGMCopula(1,0.5)
    @test_throws ArgumentError FGMCopula(3,[-1.5,2.0,3.1,1.2])
    @test_throws ArgumentError FGMCopula(1,[0.8,0.2,0.5,0.4])
end

