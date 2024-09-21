# @testitem "SklarDist fitting" begin
    
#     using Distributions
#     using Random
#     using StableRNGs
#     rng = StableRNG(123)
#     MyD = SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta()))
#     u = rand(rng,MyD,1000)
#     rand!(MyD,u)
#     fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
#     fit(SklarDist{GaussianCopula,Tuple{LogNormal,Pareto,Beta}},u)
#     @test 1==1
#     # loglikelyhood(MyD,u)
# end