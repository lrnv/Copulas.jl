module Copulas

import Base
import Random
using Distributions
using StatsBase
# Write your package code here.
include("utils.jl")
include("Copula.jl")
include("EmpiricalCopula.jl")
include("SklarDist.jl")

include("EllipticalCopula.jl")
include("EllipticalCopulas/GaussianCopula.jl")
include("EllipticalCopulas/TCopula.jl")

include("ArchimedeanCopula.jl")
include("ArchimedeanCopulas/IndependentCopula.jl")
include("ArchimedeanCopulas/ClaytonCopula.jl")
include("ArchimedeanCopulas/JoeCopula.jl")
include("ArchimedeanCopulas/GumbelCopula.jl")
include("ArchimedeanCopulas/FranckCopula.jl")
include("ArchimedeanCopulas/AMHCopula.jl")

######### Copulas 
export EmpiricalCopula
export GaussianCopula
export TCopula
export ClaytonCopula
export AMHCopula
export FranckCopula
export GumbelCopula
export IndependentCopula

######### Distributions
export SklarDist

######### utilitiaries
export pseudos

end
