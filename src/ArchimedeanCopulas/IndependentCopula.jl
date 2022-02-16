struct IndependentCopula{d} <: ArchimedeanCopula{d} end
function Distributions._logpdf(::IndependentCopula{d},u) where d
    return all(0 .<= u .<= 1) ? 1 : 0
end
function Distributions.cdf(::IndependentCopula{d},u) where d
    return all(0 .<= u .<= 1) ? prod(u) : 0
end
ϕ(::IndependentCopula,t) = exp(-t)
ϕ⁻¹(::IndependentCopula,t) = -log(t)
τ(::IndependentCopula) = 0
