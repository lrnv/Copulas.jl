###########################################################################
#####  SurvivalCopula + SubsetCopula bindings (delegation)
###########################################################################
struct FlipDistortion{Disto} <: Distortion
    base::Disto
end
Distributions.cdf(D::FlipDistortion, u::Real) = 1 - Distributions.cdf(D.base, 1 - u)
Distributions.pdf(D::FlipDistortion, u::Real) = Distributions.pdf(D.base, 1 - u)
Distributions.quantile(D::FlipDistortion, α::Real) = 1 - Distributions.quantile(D.base, 1 - α)

## Methods moved next to SurvivalCopula type
