module Copulas

import Base
import Random
import InteractiveUtils
import SpecialFunctions
import Roots
import Distributions
import StatsBase
import StatsFuns
import TaylorSeries
import ForwardDiff
import Cubature
import MvNormalCDF
import WilliamsonTransforms
import Combinatorics
import LogExpFunctions
import QuadGK
import LinearAlgebra

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

# These three distributions might be merged in Distrbutions.jl one day.
include("UnivariateDistribution/Sibuya.jl")
include("UnivariateDistribution/Logarithmic.jl")
include("UnivariateDistribution/AlphaStable.jl")
include("UnivariateDistribution/ClaytonWilliamsonDistribution.jl")
include("UnivariateDistribution/WilliamsonFromFrailty.jl")
include("UnivariateDistribution/ExtremeDist.jl")

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
include("BivariateArchimedeanMethods.jl")
export ArchimedeanCopula,
    IndependentCopula,
    MCopula,
    WCopula,
    AMHCopula,
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
            # IndependentCopula(3),
            # AMHCopula(3, 0.6),
            # AMHCopula(4, -0.3),
            # ClaytonCopula(2, -0.7),
            # ClaytonCopula(3, -0.1),
            # ClaytonCopula(4, 7),
            # FrankCopula(2, -5),
            # FrankCopula(3, 12),
            # JoeCopula(3, 7),
            GumbelCopula(4, 7),
            # GumbelBarnettCopula(3, 0.7),
            # InvGaussianCopula(4, 0.05),
            # InvGaussianCopula(3, 8),
            GaussianCopula([1 0.5; 0.5 1]),
            TCopula(4, [1 0.5; 0.5 1]),
            FGMCopula(2, 1),
            # MCopula(4),
            # ArchimedeanCopula(2, Copulas.i𝒲(Distributions.LogNormal(), 2)),
            PlackettCopula(2.0),
            EmpiricalCopula(randn(2, 100), pseudo_values=false),
            SurvivalCopula(ClaytonCopula(2, -0.7), (1, 2)),
            # WCopula(2),            ################ <<<<<<<<<-------------- Does not work and I cannot explain why !
            # RafteryCopula(2, 0.2), ################ <<<<<<<<<<------------- BUGGY
            # RafteryCopula(3, 0.5), ################ <<<<<<<<<<------------- BUGGY
            # We should probably add others to speed up again.
        )
            u1 = rand(C)
            u = rand(C, 2)
            if applicable(Distributions.pdf, C, u1) && !(typeof(C) <: EmpiricalCopula)
                Distributions.pdf(C, u1)
                Distributions.pdf(C, u)
            end
            Distributions.cdf(C, u1)
            Distributions.cdf(C, u)
        end
    end
end

end
