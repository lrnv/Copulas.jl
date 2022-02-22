abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.length(::Copula{d}) where d = d

# The potential functions to code: 
# Distributions._logpdf
# Distributions.cdf
# Distributions.fit(::Type{CT},u) where CT<:Mycopula
# Distributions._rand!
# Base.rand
# Base.eltype
# τ, τ⁻¹
# Base.eltype 






