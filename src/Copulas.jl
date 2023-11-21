module Copulas

    import Base
    import Random
    import SpecialFunctions
    import Roots
    import Distributions
    import StatsBase
    import TaylorSeries
    import ForwardDiff
    import Cubature
    import MvNormalCDF
    import WilliamsonTransforms
    import Combinatorics
    import LogExpFunctions
    import QuadGK

    # Standard copulas and stuff. 
    include("utils.jl")
    include("Copula.jl")
    include("SklarDist.jl")
    export pseudos, 
           EmpiricalCopula, 
           SklarDist

    # Others. 
    include("MiscellaneousCopulas/SurvivalCopula.jl")
    include("MiscellaneousCopulas/PlackettCopula.jl")
    include("MiscellaneousCopulas/EmpiricalCopula.jl")
    include("MiscellaneousCopulas/FGMCopula.jl")
    export SurvivalCopula,
           PlackettCopula,
           EmpiricalCopula,
           FGMCopula

    # Elliptical copulas
    include("EllipticalCopula.jl")
    include("EllipticalCopulas/GaussianCopula.jl")
    include("EllipticalCopulas/TCopula.jl")
    export GaussianCopula, 
           TCopula

    # These three distributions might be merged in Distrbutions.jl one day. 
    include("UnivariateDistribution/Sibuya.jl")
    include("UnivariateDistribution/Logarithmic.jl")
    include("UnivariateDistribution/AlphaStable.jl")
    include("UnivariateDistribution/ClaytonWilliamsonDistribution.jl")
    include("UnivariateDistribution/WilliamsonFromFrailty.jl")

    # Archimedean generators
    include("Generator.jl")

    include("Generator/WilliamsonGenerator.jl")
    export WilliamsonGenerator, i𝒲

    include("Generator/ZeroVariateGenerator/IndependentGenerator.jl")
    include("Generator/ZeroVariateGenerator/MGenerator.jl")
    include("Generator/ZeroVariateGenerator/WGenerator.jl")
    include("Generator/UnivariateGenerator/AMHGenerator.jl")
    include("Generator/UnivariateGenerator/ClaytonGenerator.jl")
    include("Generator/UnivariateGenerator/FrankGenerator.jl")
    include("Generator/UnivariateGenerator/GumbelBarnettGenerator.jl")
    include("Generator/UnivariateGenerator/GumbelGenerator.jl")
    include("Generator/UnivariateGenerator/InvGaussianGenerator.jl")
    include("Generator/UnivariateGenerator/JoeGenerator.jl")
    
    # Archimedean copulas
    include("ArchimedeanCopula.jl")
    export ArchimedeanCopula,
           IndependentCopula, 
           ClaytonCopula,
           JoeCopula,
           GumbelCopula,
           FrankCopula,
           AMHCopula,
           GumbelBarnettCopula,
           InvGaussianCopula,
           MCopula,
           WCopula

end
