abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.length(::Copula{d}) where d = d





