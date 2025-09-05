module Copulas

    import Base
    import Random
    import InteractiveUtils
    import SpecialFunctions
    import Roots
    import Distributions
    import StatsBase
    import StatsFuns
    import ForwardDiff
    import HCubature
    import MvNormalCDF
    import WilliamsonTransforms
    import Combinatorics
    import LogExpFunctions
    import QuadGK
    import LinearAlgebra
    import PolyLog
    import BigCombinatorics
    import LambertW

    # Standard copulas and stuff.
    include("utils.jl")
    include("Copula.jl")
    include("SklarDist.jl")
    export pseudos,
           SklarDist

    # Others.
    include("MiscellaneousCopulas/SurvivalCopula.jl")
    include("MiscellaneousCopulas/PlackettCopula.jl")
    include("MiscellaneousCopulas/EmpiricalCopula.jl")
    include("MiscellaneousCopulas/FGMCopula.jl")
    include("MiscellaneousCopulas/RafteryCopula.jl")
    export SurvivalCopula,
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


    include("MiscellaneousCopulas/IndependentCopula.jl")
    include("MiscellaneousCopulas/MCopula.jl")
    include("MiscellaneousCopulas/WCopula.jl")
    export IndependentCopula, 
           MCopula, 
           WCopula

    # These three distributions might be merged in Distrbutions.jl one day.
    include("UnivariateDistribution/Sibuya.jl")
    include("UnivariateDistribution/Logarithmic.jl")
    include("UnivariateDistribution/AlphaStable.jl")
    include("UnivariateDistribution/ClaytonWilliamsonDistribution.jl")
    include("UnivariateDistribution/WilliamsonFromFrailty.jl")
    include("UnivariateDistribution/ExtremeDist.jl")
    include("UnivariateDistribution/PStable.jl")
    include("UnivariateDistribution/TiltedPositiveStable.jl")
    include("UnivariateDistribution/PosStableStoppedGamma.jl")
    include("UnivariateDistribution/GammaStoppedGamma.jl")
    include("UnivariateDistribution/GammaStoppedPositiveStable.jl")
    include("UnivariateDistribution/SibuyaStoppedGamma.jl")
    include("UnivariateDistribution/SibuyaStoppedPosStable.jl")
    include("UnivariateDistribution/GeneralizedSibuya.jl")
    include("UnivariateDistribution/ShiftedNegBin.jl")

    # Archimedean generators
    include("Generator.jl")
    include("ArchimedeanCopula.jl")
    include("Generator/FrailtyGenerator.jl")
    
    include("Generator/AMHGenerator.jl")
    include("Generator/BB1Generator.jl")
    include("Generator/BB2Generator.jl")
    include("Generator/BB3Generator.jl")
    include("Generator/BB6Generator.jl")
    include("Generator/BB7Generator.jl")
    include("Generator/BB8Generator.jl")
    include("Generator/BB9Generator.jl")
    include("Generator/BB10Generator.jl")
    include("Generator/ClaytonGenerator.jl")
    include("Generator/FrankGenerator.jl")
    include("Generator/GumbelBarnettGenerator.jl")
    include("Generator/GumbelGenerator.jl")
    include("Generator/InvGaussianGenerator.jl")
    include("Generator/JoeGenerator.jl")
    include("Generator/WilliamsonGenerator.jl")
    export WilliamsonGenerator, 
           i𝒲, 
           ArchimedeanCopula,
           AMHCopula,
           BB1Copula, 
           BB2Copula, 
           BB3Copula, 
           BB6Copula, 
           BB7Copula, 
           BB8Copula, 
           BB9Copula, 
           BB10Copula,
           ClaytonCopula,
           FrankCopula,
           GumbelBarnettCopula,
           GumbelCopula,
           InvGaussianCopula,
           JoeCopula

    # bivariate Extreme Value Copulas
    include("ExtremeValueCopula.jl")
    include("ExtremeValueCopulas/AsymGalambosCopula.jl")
    include("ExtremeValueCopulas/AsymLogCopula.jl")
    include("ExtremeValueCopulas/AsymMixedCopula.jl")
    include("ExtremeValueCopulas/BC2Copula.jl")
    include("ExtremeValueCopulas/CuadrasAugeCopula.jl")
    include("ExtremeValueCopulas/GalambosCopula.jl")
    include("ExtremeValueCopulas/HuslerReissCopula.jl")
    include("ExtremeValueCopulas/LogCopula.jl")
    include("ExtremeValueCopulas/MixedCopula.jl")
    include("ExtremeValueCopulas/MOCopula.jl")
    include("ExtremeValueCopulas/tEVCopula.jl")

    export AsymGalambosCopula,
           AsymLogCopula,
           AsymMixedCopula,
           BC2Copula,
           CuadrasAugeCopula,
           GalambosCopula,
           HuslerReissCopula,
           LogCopula,
           MixedCopula,
           MOCopula,
           tEVCopula

    include("ArchimaxCopula.jl")
    include("MiscellaneousCopulas/BB4Copula.jl")
    include("MiscellaneousCopulas/BB5Copula.jl")

    export ArchimaxCopula,
           BB4Copula,
           BB5Copula

    # Subsetting
    include("SubsetCopula.jl") # not exported yet.

    # transformations
    export rosenblatt, inverse_rosenblatt

    using PrecompileTools
    @setup_workload begin
        # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
        # precompile file and potentially make loading faster.
        @compile_workload begin
            for C in (
                IndependentCopula(3),
                AMHCopula(3,0.6),
                AMHCopula(4,-0.01),
                ClaytonCopula(2,-0.7),
                ClaytonCopula(3,-0.1),
                ClaytonCopula(4,7),
                FrankCopula(2,-5),
                FrankCopula(3,12),
                JoeCopula(3,7),
                GumbelCopula(4,7),
                GaussianCopula([1 0.5; 0.5 1]),
                TCopula(4, [1 0.5; 0.5 1]),
                FGMCopula(2,1),
            )
                u1 = rand(C)
                u = rand(C,2)
                if applicable(Distributions.pdf,C,u1)
                    Distributions.pdf(C,u1)
                    Distributions.pdf(C,u)
                end
                Distributions.cdf(C,u1)
                Distributions.cdf(C,u)
            end
        end
    end
end
