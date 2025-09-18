"""
    PlackettCopula(θ)

Symmetric one–parameter bivariate copula allowing both positive (`θ>1`) and negative (`0<θ<1`) dependence, useful for quick benchmarking as a simple non‑Archimedean, non‑elliptical family with closed‑form CDF and density. Parameter:
* `θ > 0` – dependence; boundary / singular values map to existing types:
  * `θ = 1` → `IndependentCopula(2)`
  * `θ = 0` → `MCopula(2)` (upper Fréchet–Hoeffding bound in this code base’s convention)
  * `θ = Inf` → `WCopula(2)` (lower Fréchet–Hoeffding bound in this convention)

For `θ ≠ 1` the CDF is
```math
C_\\theta(u,v) = \\frac{1 + (\\theta-1)(u+v) - \\sqrt{(1+(\\theta-1)(u+v))^2 - 4\\,u v\\,\\theta(\\theta-1)}}{2(\\theta-1)}
```
and for `θ = 1`, `C_1(u,v)=uv`. The density for `θ ≠ 1` is
```math
c_\\theta(u,v) = \\frac{\\theta (1 + (\\theta-1)(u+v-2uv))}{\\left[(1+(\\theta-1)(u+v))^2 - 4\\,\\theta(\\theta-1)uv\\right]^{3/2}}
```
Classical limit behavior is `θ→0` → lower bound `W` and `θ→∞` → upper bound `M`; the constructor mappings reverse these to preserve historical package tests. See also: [`MCopula`](@ref), [`WCopula`](@ref), [`IndependentCopula`](@ref).

References
* [joe2014](@cite) Joe (2014) Dependence Modeling with Copulas, p.164.
* [johnson1987multivariate](@cite) Johnson (1987) Multivariate Statistical Simulation, p.193.
* [nelsen2006](@cite) Nelsen (2006) *An Introduction to Copulas*, Ex. 3.38.
"""
struct PlackettCopula{P} <: Copula{2} # since it is only bivariate.
    θ::P  # Copula parameter

    function PlackettCopula(θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative"))
        elseif θ == 0
            return MCopula(2)
        elseif θ == 1
            return IndependentCopula(2)
        elseif θ == Inf
            return WCopula(2)
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end

Base.eltype(S::PlackettCopula{P}) where {P} = P # this shuold be P. 

# CDF calculation for bivariate Plackett Copula
function _cdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = 1 + η * (u + v)
    term2 = sqrt(term1^2 - 4 * S.θ * η * u * v)
    return 0.5 * η^(-1) * (term1 - term2)
end

# PDF calculation for bivariate Plackett Copula
function Distributions._logpdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = S.θ * (1 + η * (u + v - 2 * u * v))
    term2 = (1+η*(u+v))^2-4*(S.θ)*η*u*v
    return log(term1) - 3 * log(term2)/2 # since we are supposed to return the logpdf. 
end
import Random

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:PlackettCopula}
    u = rand(rng)
    t = rand(rng)
    a = t * (1 - t)
    b = C.θ + a * (C.θ - 1)^2
    cc = 2a * (u * C.θ^2 + 1 - u) + C.θ * (1 - 2a)
    d = sqrt(C.θ) * sqrt(C.θ + 4a * u * (1 - u) * (1 - C.θ)^2)
    v = (cc - (1 - 2t) * d) / (2b)
    x[1] = u
    x[2] = v
    return x
end

# Calculate Spearman's rho based on the PlackettCopula parameters
function ρ(c::PlackettCopula{P}) where P
    return (c.θ+1)/(c.θ-1)-(2*c.θ*log(c.θ)/(c.θ-1)^2)
end

# Conditioning colocated
function DistortionFromCop(C::PlackettCopula, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int)
    return PlackettDistortion(float(C.θ), Int8(js[1]), float(uⱼₛ[1]))
end