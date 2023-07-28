module Copulas


@doc let
       path = joinpath(dirname(@__DIR__), "README.md")
       include_dependency(path)
       read(path, String)
end Copulas

    import Base
    import Random
    import SpecialFunctions
    import GSL
    import Roots
    using Distributions
    using StatsBase
    using TaylorSeries
    import ForwardDiff
    import Cubature
    import MvNormalCDF
    import AlphaStableDistributions
    using PrecompileTools 

    # Standard copulas and stuff. 
    include("utils.jl")
    include("Copula.jl")
    include("EmpiricalCopula.jl")
    include("SklarDist.jl")
    export pseudos, 
           EmpiricalCopula, 
           SklarDist

    # These three distributions might be merged in Distrbutions.jl one day. 
    include("univariate_distributions/Sibuya.jl")
    include("univariate_distributions/Logarithmic.jl")

    # Elliptical copulas
    include("EllipticalCopula.jl")
    include("EllipticalCopulas/GaussianCopula.jl")
    include("EllipticalCopulas/TCopula.jl")
    export GaussianCopula, 
           TCopula

    # Archimedean copulas
    include("ArchimedeanCopula.jl")
    include("ArchimedeanCopulas/IndependentCopula.jl")
    include("ArchimedeanCopulas/ClaytonCopula.jl")
    include("ArchimedeanCopulas/JoeCopula.jl")
    include("ArchimedeanCopulas/GumbelCopula.jl")
    include("ArchimedeanCopulas/FrankCopula.jl")
    include("ArchimedeanCopulas/AMHCopula.jl")
    export IndependentCopula, 
           ClaytonCopula,
           JoeCopula,
           GumbelCopula,
           FrankCopula,
           AMHCopula

    # Others. 
    include("MiscellaneousCopulas/MCopula.jl")
    include("MiscellaneousCopulas/WCopula.jl")
    include("MiscellaneousCopulas/SurvivalCopula.jl")
    export MCopula,
        Wcopula,
        SurvivalCopula

     # PrecompileTools stuff
     @setup_workload begin
       biv_cops = [GaussianCopula([1 0.7; 0.7 1]), TCopula(2,[1 0.7; 0.7 1]),ClaytonCopula(2,7),JoeCopula(2,3),GumbelCopula(2,8),FrankCopula(2,0.5), AMHCopula(2,0.7)]
       @compile_workload begin
           for C in biv_cops
              u = Random.rand(C,10)
              pdf(C,[0.5,0.5])
              cdf(C,[0.5,0.5])
              D = SklarDist(C,[Gamma(1,1),Normal(1,1)])
              u = Random.rand(D,10)
              pdf(D,[0.5,0.5])
              cdf(D,[0.5,0.5])
           end
       end
   end

end
