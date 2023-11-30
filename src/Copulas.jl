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
    include("MiscellaneousCopulas/MCopula.jl")
    include("MiscellaneousCopulas/WCopula.jl")
    include("MiscellaneousCopulas/SurvivalCopula.jl")
    include("MiscellaneousCopulas/PlackettCopula.jl")
    include("MiscellaneousCopulas/EmpiricalCopula.jl")
    include("MiscellaneousCopulas/FGMCopula.jl")
    include("MiscellaneousCopulas/RafteryCopula.jl")
    export MCopula,
           WCopula,
           SurvivalCopula,
           PlackettCopula,
           EmpiricalCopula,
           FGMCopula,
           RafteryCopula

    # Elliptical copulas
    include("EllipticalCopula.jl")
    include("EllipticalCopulas/GaussianCopula.jl")
    include("EllipticalCopulas/TCopula.jl")
    export GaussianCopula, 
           TCopula

    # These three distributions might be merged in Distrbutions.jl one day. 
    include("univariate_distributions/Sibuya.jl")
    include("univariate_distributions/Logarithmic.jl")
    include("univariate_distributions/AlphaStable.jl")
    include("univariate_distributions/ClaytonWilliamsonDistribution.jl")
    include("univariate_distributions/WilliamsonFromFrailty.jl")

    # Archimedean copulas
    include("ArchimedeanCopula.jl")
    include("ArchimedeanCopulas/IndependentCopula.jl")
    include("ArchimedeanCopulas/ClaytonCopula.jl")
    include("ArchimedeanCopulas/JoeCopula.jl")
    include("ArchimedeanCopulas/GumbelCopula.jl")
    include("ArchimedeanCopulas/FrankCopula.jl")
    include("ArchimedeanCopulas/AMHCopula.jl")
    include("ArchimedeanCopulas/WilliamsonCopula.jl")
    include("ArchimedeanCopulas/GumbelBarnettCopula.jl")
    include("ArchimedeanCopulas/InvGaussianCopula.jl")
    export IndependentCopula, 
           ClaytonCopula,
           JoeCopula,
           GumbelCopula,
           FrankCopula,
           AMHCopula,
           WilliamsonCopula,
           GumbelBarnettCopula,
           InvGaussianCopula

end
