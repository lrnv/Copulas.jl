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
    include("Subsetting.jl")
    export pseudos,
           SklarDist,
           rosenblatt,
           inverse_rosenblatt, 
           subsetdims

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

    # Frailties
    include("UnivariateDistribution/Frailties/Sibuya.jl")
    include("UnivariateDistribution/Frailties/Logarithmic.jl")
    include("UnivariateDistribution/Frailties/AlphaStable.jl")
    include("UnivariateDistribution/Frailties/GammaStoppedGamma.jl")
    include("UnivariateDistribution/Frailties/GammaStoppedPositiveStable.jl")
    include("UnivariateDistribution/Frailties/PosStableStoppedGamma.jl")
    include("UnivariateDistribution/Frailties/SibuyaStoppedGamma.jl")
    include("UnivariateDistribution/Frailties/SibuyaStoppedPosStable.jl")
    include("UnivariateDistribution/Frailties/GeneralizedSibuya.jl")
    include("UnivariateDistribution/Frailties/ShiftedNegBin.jl")

    # Radials (Williamson transforms etc.)
    include("UnivariateDistribution/Radials/ExtremeDist.jl")
    include("UnivariateDistribution/Radials/PStable.jl")
    include("UnivariateDistribution/Radials/TiltedPositiveStable.jl")
    include("UnivariateDistribution/Radials/ClaytonWilliamsonDistribution.jl")
    include("UnivariateDistribution/Radials/WilliamsonFromFrailty.jl")

    # Archimedean mechanics
    include("Generator.jl")
    include("ArchimedeanCopula.jl")
    include("Generator/FrailtyGenerator.jl")
    
    # Archimedean Generators
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

    # Bivariate Extreme Value Copulas
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

    # Conditional distributions (uniform/original scales)
    include("Conditioning.jl")
    include("UnivariateDistribution/Distortions/NoDistortion.jl")
    include("UnivariateDistribution/Distortions/GaussianDistortion.jl")
    include("UnivariateDistribution/Distortions/StudentDistortion.jl")
    include("UnivariateDistribution/Distortions/BivEVDistortion.jl")
    include("UnivariateDistribution/Distortions/PlackettDistortion.jl")
    include("UnivariateDistribution/Distortions/BivFGMDistortion.jl")
    include("UnivariateDistribution/Distortions/MDistortion.jl")
    include("UnivariateDistribution/Distortions/WDistortion.jl")
    include("UnivariateDistribution/Distortions/ArchimedeanDistortion.jl")
    include("UnivariateDistribution/Distortions/FlipDistortion.jl")
    export condition

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
