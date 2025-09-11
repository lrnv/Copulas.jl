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

    # Main code
    include("utils.jl")
    include("Copula.jl")
    include("SklarDist.jl")
    include("Subsetting.jl")
    include("Conditioning.jl")

    # Frailties (Univ r.v. on R_+ which Laplace transform are used as arch. generators)
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

    # Radials (Univ r.v. on R_+ which Williamson d-transform are used as arch. generators)
    include("UnivariateDistribution/Radials/PStable.jl")
    include("UnivariateDistribution/Radials/TiltedPositiveStable.jl")
    include("UnivariateDistribution/Radials/ClaytonWilliamsonDistribution.jl")
    include("UnivariateDistribution/Radials/WilliamsonFromFrailty.jl")

    # Distortions (Univ r.v. on [0,1] which are conditional distributions from copulas)
    include("UnivariateDistribution/Distortions/NoDistortion.jl")
    include("UnivariateDistribution/Distortions/GaussianDistortion.jl")
    include("UnivariateDistribution/Distortions/StudentDistortion.jl")
    include("UnivariateDistribution/Distortions/BivEVDistortion.jl")
    include("UnivariateDistribution/Distortions/PlackettDistortion.jl")
    include("UnivariateDistribution/Distortions/BivFGMDistortion.jl")
    include("UnivariateDistribution/Distortions/BivArchimaxDistortion.jl")
    include("UnivariateDistribution/Distortions/MDistortion.jl")
    include("UnivariateDistribution/Distortions/WDistortion.jl")
    include("UnivariateDistribution/Distortions/FlipDistortion.jl")
    include("UnivariateDistribution/Distortions/ArchimedeanDistortion.jl")

    # Others, usefull too 
    include("UnivariateDistribution/ExtremeDist.jl")

    # Miscelaneous copulas
    include("MiscellaneousCopulas/SurvivalCopula.jl")
    include("MiscellaneousCopulas/PlackettCopula.jl")
    include("MiscellaneousCopulas/EmpiricalCopula.jl")
    include("MiscellaneousCopulas/FGMCopula.jl")
    include("MiscellaneousCopulas/RafteryCopula.jl")
    include("MiscellaneousCopulas/IndependentCopula.jl")
    include("MiscellaneousCopulas/MCopula.jl")
    include("MiscellaneousCopulas/WCopula.jl")

    # Elliptical copulas
    include("EllipticalCopula.jl")
    include("EllipticalCopulas/GaussianCopula.jl")
    include("EllipticalCopulas/TCopula.jl")

    # Archimedean copulas
    include("Generator.jl")
    include("ArchimedeanCopula.jl")

    # Generators
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
    include("Generator/PowerGenerator.jl")

    #Extreme value copulas
    include("Tail.jl")
    include("ExtremeValueCopula.jl")

    # Stable tail dependence functions
    include("Tail/NoTail.jl")
    include("Tail/MTail.jl")
    include("Tail/AsymGalambosTail.jl")
    include("Tail/AsymLogTail.jl")
    include("Tail/AsymMixedTail.jl")
    include("Tail/BC2Tail.jl")
    include("Tail/CuadrasAugeTail.jl")
    include("Tail/GalambosTail.jl")
    include("Tail/HuslerReissTail.jl")
    include("Tail/LogTail.jl")
    include("Tail/MixedTail.jl")
    include("Tail/MOTail.jl")
    include("Tail/tEVTail.jl")

    # Archimax copulas (includes the BB4 and BB5 models)
    include("ArchimaxCopula.jl")

    # overloadings of the show() function: 
    include("show.jl")

    export pseudos, # utility functions and methods making the interface: 
           rosenblatt, 
           inverse_rosenblatt, 
           subsetdims, 
           condition,
           WilliamsonGenerator, 
           i𝒲, 
           TiltedGenerator,
           PowerGenerator,
           SklarDist, # SklarDist to make multivariate models
           AMHCopula, # And a bunch of copulas. 
           ArchimedeanCopula,
           AsymGalambosCopula,
           AsymLogCopula,
           AsymMixedCopula,
           BB10Copula,
           BB1Copula, 
           BB2Copula, 
           BB3Copula, 
           BB6Copula, 
           BB7Copula, 
           BB8Copula, 
           BB9Copula, 
           BC2Copula,
           ClaytonCopula,
           CuadrasAugeCopula,
           EmpiricalCopula,
           FGMCopula,
           FrankCopula,
           GalambosCopula,
           GaussianCopula,
           GumbelBarnettCopula,
           GumbelCopula,
           HuslerReissCopula,
           IndependentCopula, 
           InvGaussianCopula,
           JoeCopula,
           LogCopula,
           MCopula, 
           MixedCopula,
           MOCopula,
           PlackettCopula,
           RafteryCopula,
           SurvivalCopula,
           TCopula,
           tEVCopula,
           WCopula,
           ArchimaxCopula,
           BB4Copula,
           BB5Copula

end
