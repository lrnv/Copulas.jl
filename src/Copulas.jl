module Copulas

    import Base
    import Random
    import SpecialFunctions
    import Roots
    import Distributions
    import Statistics
    import StatsBase
    import StatsFuns
    import ForwardDiff
    import HCubature
    import MvNormalCDF
    import Combinatorics
    import LogExpFunctions
    import QuadGK
    import LinearAlgebra
    import PolyLog
    import LambertW
    import Optim
    import Printf
    import TaylorSeries
    import ADTypes

    # Main code
    include("utils.jl")
    include("Copula.jl")
    include("SklarDist.jl")
    include("Subsetting.jl")
    include("Conditioning.jl")
    include("Fitting.jl")
    include("Nataf.jl")

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
    include("UnivariateDistribution/Distortions/HistogramDistortion.jl")
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
    include("LiouvilleCopula.jl")

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

    # Nested (hierarchical) Archimedean copulas
    include("NestedArchimedeanCopula.jl")

    #Extreme value copulas
    include("Tail.jl")
    include("Tail/utilities.jl")
    include("ExtremeValueCopula.jl")

    # Stable tail dependence functions
    include("Tail/NoTail.jl")
    include("Tail/MTail.jl")
    include("Tail/AsymGalambosTail.jl")
    include("Tail/AsymLogTail.jl")
    include("Tail/TawnTail.jl")
    include("Tail/AsymMixedTail.jl")
    include("Tail/DiscreteSpectralTail.jl")
    include("Tail/BC2Tail.jl")
    include("Tail/CuadrasAugeTail.jl")
    include("Tail/GalambosTail.jl")
    include("Tail/HuslerReissTail.jl")
    include("Tail/LogTail.jl")
    include("Tail/MixedTail.jl")
    include("Tail/MOTail.jl")
    include("Tail/tEVTail.jl")
    include("Tail/EmpiricalEVTail.jl")

    include("MiscellaneousCopulas/BernsteinCopula.jl")
    include("MiscellaneousCopulas/BetaCopula.jl")
    include("MiscellaneousCopulas/CheckerboardCopula.jl")
    # Archimax copulas (includes the BB4 and BB5 models)
    include("ArchimaxCopula.jl")


    include("show.jl")

    export pseudos, condition, subsetdims, rosenblatt, inverse_rosenblatt, Nataf
    export SklarDist, CopulaModel

    export WilliamsonGenerator, 𝒲, EmpiricalGenerator, DiscreteSpectralTail
    export ArchimedeanCopula, ExtremeValueCopula, LiouvilleCopula
    export NestedArchimedeanCopula, ArchimaxCopula

    export AMHCopula, ClaytonCopula, FrankCopula, GumbelCopula
    export GumbelBarnettCopula, InvGaussianCopula, JoeCopula
    export BB1Copula, BB2Copula, BB3Copula, BB6Copula, BB7Copula
    export BB8Copula, BB9Copula, BB10Copula

    export AsymGalambosCopula, AsymLogCopula, AsymMixedCopula, BC2Copula
    export CuadrasAugeCopula, EmpiricalEVCopula, GalambosCopula
    export HuslerReissCopula, LogCopula, MixedCopula, MOCopula
    export TawnCopula, tEVCopula, BB4Copula, BB5Copula

    export GaussianCopula, TCopula
    export BernsteinCopula, BetaCopula, CheckerboardCopula, EmpiricalCopula
    export FGMCopula, IndependentCopula, MCopula, WCopula
    export PlackettCopula, RafteryCopula, SurvivalCopula

    public Copula, Distortion, Generator, Tail

    public ϕ, ϕ⁻¹, ϕ⁽¹⁾, ϕ⁻¹⁽¹⁾, ϕ⁽ᵏ⁾, ϕ⁽ᵏ⁾⁻¹, 𝒲₋₁, max_monotony
    public A, dA, d²A, ℓ, ellpartial

    public τ, ρ, β, γ, ι, λₗ, λᵤ
    public τ⁻¹, ρ⁻¹, β⁻¹, λᵤ⁻¹
    public corblomqvist, corgini, corentropy, corlowertail, coruppertail, measure

    public IndependentGenerator, MGenerator, WGenerator, FrailtyGenerator
    public AMHGenerator, ClaytonGenerator, FrankGenerator, GumbelGenerator
    public GumbelBarnettGenerator, InvGaussianGenerator, JoeGenerator
    public BB1Generator, BB2Generator, BB3Generator, BB6Generator, BB7Generator
    public BB8Generator, BB9Generator, BB10Generator

    public AsymGalambosTail, AsymLogTail, AsymMixedTail, BC2Tail, CuadrasAugeTail
    public EmpiricalEVTail, EmpiricalEVMultivariateTail, GalambosTail
    public HuslerReissTail, LogTail, MixedTail, MOTail, TawnTail, tEVTail

end
