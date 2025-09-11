"""
    ClaytonGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonGenerator(θ)
    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)`` (with the independence case as the limit ``\\theta\\to 0``). It is an Archimedean copula with generator

```math
\\phi(t) = \\left(1 + \\theta t\\right)^{-1/\\theta}
```

with the continuous extension ``\\phi(t) = e^{-t}`` at ``\\theta = 0``.

Special cases (for the copula in dimension ``d``):
- When ``\\theta = -1/(d-1)``, it is the WCopula (Lower Fréchet–Hoeffding bound)
- When ``\\theta \\to 0``, it is the IndependentCopula
- When ``\\theta \\to \\infty``, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct ClaytonGenerator{T} <: Generator
    θ::T
    function ClaytonGenerator(θ)
        if θ < -1
            throw(ArgumentError("Theta must be greater than -1"))
        elseif θ == -1
            return WGenerator()
        elseif θ == 0
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const ClaytonCopula{d, T} = ArchimedeanCopula{d, ClaytonGenerator{T}}
ClaytonCopula(d, θ) = ArchimedeanCopula(d, ClaytonGenerator(θ))
Distributions.params(G::ClaytonGenerator) = (G.θ,)


max_monotony(G::ClaytonGenerator) = G.θ >= 0 ? Inf : Int(floor(1 - 1/G.θ))
ϕ(  G::ClaytonGenerator, t) = max(1+G.θ*t,zero(t))^(-1/G.θ)
ϕ⁻¹(G::ClaytonGenerator, t) = (t^(-G.θ)-1)/G.θ
ϕ⁽¹⁾(G::ClaytonGenerator, t) = (1+G.θ*t) ≤ 0 ? 0 : - (1+G.θ*t)^(-1/G.θ -1)
ϕ⁻¹⁽¹⁾(G::ClaytonGenerator, t) = -t^(-G.θ-1)
ϕ⁽ᵏ⁾(G::ClaytonGenerator, ::Val{k}, t) where k = (1+G.θ*t) ≤ 0 ? 0 : (1 + G.θ * t)^(-1/G.θ - k) * prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1)
ϕ⁽ᵏ⁾⁻¹(G::ClaytonGenerator, ::Val{k}, t; start_at=t) where k = ((t / prod(-1-ℓ*G.θ for ℓ in 0:k-1; init=1))^(1/(-1/G.θ - k)) -1)/G.θ    

τ(G::ClaytonGenerator) = ifelse(isfinite(G.θ), G.θ/(G.θ+2), 1)
τ⁻¹(::Type{T},τ) where T<:ClaytonGenerator = ifelse(τ == 1,Inf,2τ/(1-τ))
williamson_dist(G::ClaytonGenerator, ::Val{d}) where d = G.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/G.θ,G.θ), Val{d}()) : ClaytonWilliamsonDistribution(G.θ,d)

frailty(G::ClaytonGenerator) = G.θ >= 0 ? Distributions.Gamma(1/G.θ, G.θ) : throw(ArgumentError("Clayton frailty is only defined for θ ≥ 0 (positive dependence). Got θ = $(G.θ)."))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ClaytonCopula, A::DenseMatrix{<:Real})
    A[:] = inverse_rosenblatt(C, rand(rng, size(A)...))
    return A
end
function Distributions._logpdf(C::ClaytonCopula{d,TG}, u) where {d,TG<:ClaytonGenerator}
    # Check if all elements are in (0,1) and if θ < 0, check the sum condition
    if !all(0 .< u .< 1) || (C.G.θ < 0 && sum(u .^ -(C.G.θ)) < (d - 1))
        return eltype(u)(-Inf)
    end

    θ = C.G.θ
    # Compute the sum of transformed variables
    S1 = sum(t ^ (-θ) for t in u)
    S2 = sum(log(t) for t in u)
    # Compute the log of the density according to the explicit formula for Clayton copula
    # See McNeil & Neslehova (2009), eq. (13)
    S1==d-1 && return eltype(u)(-Inf)
    return log(θ + 1) * (d - 1) - (θ + 1) * S2 + (-1 / θ - d) * log(S1 - d + 1)
end