"""
    Sibuya(p)

Parameters
    * `p ∈ (0,1]` – tail parameter (smaller ⇒ heavier tail)

Used as frailty to construct Archimedean copulas via the Laplace transform
`ϕ(t)=E[e^{-t X}] = 1 - (1 - e^{-t})^{p}` (see [`FrailtyGenerator`](@ref)).

Sibuya discrete distribution (heavy–tailed).

Definition
Support: k = 1,2,3, …  (code keeps `minimum(::Sibuya)=0` for legacy reasons; mass at 0 is 0.)

One convenient characterization is by its cumulative distribution function

```math
F(k) = 1 - \\left|\\binom{p-1}{k}\\right|, \\quad k \\ge 0, \\qquad (F(-1)=0),
```
so the tail probabilities satisfy `P(X > k) = |binom(p-1,k+1)|`.

Its moment generating function (finite for t ≤ 0) is implemented as

```math
M_X(t) = 1 - (1 - e^{t})^{p},  \\qquad t \\le 0.
```

(The pmf has no simple closed form involving only elementary functions; it can
be expressed via forward differences of the generalized binomial coefficients.)

Uses
* As a frailty: `ϕ(t) = E[e^{-tX}] = 1 - (1 - e^{-t})^{p}` serves as an Archimedean generator.

See also: [`FrailtyGenerator`](@ref), other frailties in `Frailties/`.
References: R `copula::Sibuya` documentation; standard treatments of Sibuya laws : https://rdrr.io/rforge/copula/man/Sibuya.html
"""
struct Sibuya{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    p::T
    function Sibuya(p::T) where {T <: Real}
        @assert 0 < p ≤ 1
        new{T}(p)
    end
    Sibuya{T}(p) where T = Sibuya(T(p))
end
Base.minimum(::Sibuya) = 0
Base.maximum(::Sibuya) = Inf
function Distributions.rand(rng::Distributions.AbstractRNG, d::Sibuya{T}) where {T <: Real}
    u = rand(rng, T)
    if u <= d.p
        return T(1)
    end
    xMax = 1/eps(T)
    Ginv = ((1-u)*SpecialFunctions.gamma(1-d.p))^(-1/d.p)
    fGinv = floor(Ginv)
    if Ginv > xMax 
        return fGinv
    end
    if 1-u < 1/(fGinv*SpecialFunctions.beta(fGinv,1-d.p))
        return ceil(Ginv)
    end
    return fGinv
end
Distributions.mgf(D::Sibuya, t) = 1-(-expm1(t))^(D.p)
function Distributions.cdf(d::Sibuya, u::Real)
    k = trunc(u)
    return 1 - abs(binom(d.p-1, k))
end
function Distributions.logpdf(d::Sibuya, x::Real)
    insupport(d, x) ? log(abs(binom(d.p, k))) : -Inf
end