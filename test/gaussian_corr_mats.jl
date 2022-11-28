@testitem "error on corr mats" begin
    using LinearAlgebra
    @test_throws LinearAlgebra.PosDefException GaussianCopula([1 2.0; 2 1])
    @test_throws LinearAlgebra.PosDefException TCopula(10,[1 2.0; 2 1])
end