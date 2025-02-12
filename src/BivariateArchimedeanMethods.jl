# The Gumbel copula needs a specific bivariate method to handle large parameters.
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:GumbelGenerator}
    θ = C.G.θ
    x₁, x₂ = -log(u[1]), -log(u[2])
    lx₁, lx₂ = log(x₁), log(x₂)
    return 1 - LogExpFunctions.cexpexp(LogExpFunctions.logaddexp(θ * lx₁, θ * lx₂) / θ)
end
function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u::AbstractArray) where {G<:GumbelGenerator}
    T = promote_type(Float64, eltype(u))
    !all(0 .< u .<= 1) && return T(-Inf) # if not in range return -Inf

    θ = C.G.θ
    x₁, x₂ = -log(u[1]), -log(u[2])
    lx₁, lx₂ = log(x₁), log(x₂)
    A = LogExpFunctions.logaddexp(θ * lx₁, θ * lx₂)
    B = exp(A / θ)
    return -B + x₁ + x₂ + (θ - 1) * (lx₁ + lx₂) + A / θ - 2A + log(B + θ - 1)
end
