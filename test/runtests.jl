using Copulas
using Test
using Distributions
using Random

@testset "Copulas.jl" begin
    MyD = SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta()))
    u = rand(MyD,10000)
    rand!(MyD,u)
    fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
    fit(SklarDist{GaussianCopula,Tuple{LogNormal,Pareto,Beta}},u)
end
