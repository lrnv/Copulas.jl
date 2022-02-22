module Copulas

import Base
import Random
import SpecialFunctions
import GSL
import Roots
using Distributions
using StatsBase
# Write your package code here.
include("utils.jl")
include("Copula.jl")
include("EmpiricalCopula.jl")
include("SklarDist.jl")

# These three distributions might be merged in Distrbutions.jl one day. 
include("univariate_distributions/Sibuya.jl")
include("univariate_distributions/Logarithmic.jl")
include("univariate_distributions/AlphaStable.jl")

include("EllipticalCopula.jl")
include("EllipticalCopulas/GaussianCopula.jl")
include("EllipticalCopulas/TCopula.jl")

include("ArchimedeanCopula.jl")
include("ArchimedeanCopulas/IndependentCopula.jl")
include("ArchimedeanCopulas/ClaytonCopula.jl")
include("ArchimedeanCopulas/JoeCopula.jl")
include("ArchimedeanCopulas/GumbelCopula.jl")
include("ArchimedeanCopulas/FrankCopula.jl")
include("ArchimedeanCopulas/AMHCopula.jl")

######### Copulas 
export EmpiricalCopula
export GaussianCopula
export TCopula
export ClaytonCopula
export AMHCopula
export JoeCopula
export FrankCopula
export GumbelCopula
export IndependentCopula

######### Distributions
export SklarDist

######### utilitiaries
export pseudos

end
