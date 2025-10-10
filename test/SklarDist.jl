
@testset "SklarDist fitting" begin
    # [GenericTests integration]: No. This exercises fitting pathways and RNG; belongs to integration tests for SklarDist rather than generic copula properties.
    
    Random.seed!(rng,123)
    MyD = SklarDist(ClaytonCopula(3,7),[LogNormal(),Pareto(),Beta()]) # with vector and not tuple as input
    u = rand(rng,MyD,1000)
    rand!(rng, MyD,u)
    fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
    @test 1==1
    # loglikelyhood(MyD,u)
end