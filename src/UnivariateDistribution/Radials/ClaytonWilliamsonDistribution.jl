"""
    ClaytonWilliamsonDistribution(θ, d)

Parameters
    * `θ < 0` – Clayton parameter (boundary `θ = -1/(d-1)` degenerates)
    * `d ≥ 2` – copula dimension

Used as radial distribution underlying the (negative‑θ) Clayton generator via its
Williamson d‑transform.

Radial distribution whose Williamson d‑transform yields (up to scaling) the
Clayton generator with negative parameter θ < 0 (specifically in the boundary
case approaching independence at θ = -1/(d-1), it degenerates at 1).

Support: [0, -1/θ]. CDF for x ∈ [0, -1/θ]:
```math
F(x) = 1 - \\sum_{k=0}^{d-1} \\binom{-1/θ}{-1/θ-k} y^{k} (1-y)^{-1/θ-k}, \\qquad y = -θ x.
```
Density (0 < x < -1/θ):
```math
f(x) = \\binom{-1/θ - 1}{d-1} y^{d-1} (1-y)^{-1/θ - d}, \\qquad y = -θ x.
```
"""
struct ClaytonWilliamsonDistribution{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    θ::T # theta is negative here. 
    d::Int # d is an integer, at least 2. 
    function ClaytonWilliamsonDistribution(θ, d)
        if θ == -1/(d-1)
            return Distributions.Dirac(1)
        end
        @assert θ < 0
        @assert d > 1
        new{typeof(θ)}(θ, d)
    end
    ClaytonWilliamsonDistribution{T}(θ, d) where T = ClaytonWilliamsonDistribution(T(θ), d)
end
Base.minimum(D::ClaytonWilliamsonDistribution) = zero(D.θ)
Base.maximum(D::ClaytonWilliamsonDistribution) = -1/D.θ
@inline _clayton_choose(a, b) = SpecialFunctions.gamma(a + 1) / (SpecialFunctions.gamma(b + 1) * SpecialFunctions.gamma(a - b + 1))
function Distributions.cdf(D::ClaytonWilliamsonDistribution, x::Real)
    θ = D.θ
    d = D.d
    x < 0 && return zero(x)
    α = -1/θ
    x >= α && return one(x)
    rez = zero(x)
    y = x/α
    for k in 0:(d-1)
        rez += _clayton_choose(α, α-k) * y^k * (1 - y)^(α-k)
    end
    return 1-rez
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::ClaytonWilliamsonDistribution)
    u = rand(rng)
    Roots.find_zero(x -> (Distributions.cdf(d,x) - u), (0.0, Inf))
end
function Distributions.pdf(D::ClaytonWilliamsonDistribution, x::Real)
    d = D.d
    α = -1/D.θ
    (0 ≥ x || x ≥ α) && return zero(x)
    y = x/α
    return _clayton_choose(α - 1, d - 1) * y^(d-1) * (1 - y)^(α - d)
end
function Distributions.logpdf(D::ClaytonWilliamsonDistribution, x::Real)
    d = D.d
    α = -1/D.θ
    (0 ≥ x || x ≥ α) && return -Inf
    y = x/α
    return log(_clayton_choose(α - 1, d - 1)) + (d-1)*log(y) + (α - d)*log1p(-y)
end
