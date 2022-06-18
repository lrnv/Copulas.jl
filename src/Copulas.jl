module Copulas
    import Base
    import Random
    import SpecialFunctions
    import GSL
    import Roots
    using Distributions
    using StatsBase
    using TaylorSeries
    import ForwardDiff

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
    include("univariate_distributions/AlphaStable.jl")

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

end
