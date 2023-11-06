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
    import Distributions
    import StatsBase
    import TaylorSeries
    import ForwardDiff
    import Cubature
    import MvNormalCDF

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
    export MCopula,
           WCopula,
           SurvivalCopula,
           PlackettCopula,
           EmpiricalCopula

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


end
