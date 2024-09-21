# @testitem "Testing survival stuff" begin
#     using Distributions
#     using StableRNGs
#     rng = StableRNG(123)
#     C = ClaytonCopula(2,3.0) # bivariate clayton with theta = 3.0
#     C90 = SurvivalCopula(C,(1,)) # flips the first dimension
#     C270 = SurvivalCopula(C,(2,)) # flips only the second dimension. 
#     C180 = SurvivalCopula(C,(1,2)) # flips both dimensions.

#     u1,u2 = rand(rng,2)
#     p = pdf(C,[u1,u2])
#     @test pdf(C90,[1-u1,u2]) == p
#     @test pdf(C270,[u1,1-u2]) == p
#     @test pdf(C180,[1-u1,1-u2]) == p

# end