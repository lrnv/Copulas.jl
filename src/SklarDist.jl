"""
    SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution

Joint distribution built from a copula `C::CT` and a tuple of marginals `m::TplMargins`
via Sklar's theorem. Provides the full `Distributions.jl` API (rand / cdf / pdf / logpdf).

Fields
  * `C::CT` – copula (`Copula{d}`)
  * `m::TplMargins` – NTuple of `d` univariate `Distributions.UnivariateDistribution`

Constructor

    SklarDist(C, m)

where `m` is a length‑`d` tuple of marginals compatible with `length(C)`.

Example
```julia
using Copulas, Distributions
X1, X2 = Gamma(2,3), LogNormal()
C = FrankCopula(2, 3.0)
D = SklarDist(C, (X1, X2))
sim = rand(D, 1_000)
D̂ = fit(SklarDist{FrankCopula,Tuple{Gamma,LogNormal}}, sim)
```

Fitting
`fit(SklarDist{CT,Tmarg}, data)` fits each marginal independently (by calling
`fit` for each declared marginal type) then fits the copula on the pseudo‑observations.

See also: [`Copula`](@ref), [`subsetdims`](@ref), [`condition`](@ref).

References
* [sklar1959](@cite) Sklar (1959).
* [nelsen2006](@cite) Nelsen (2006), *An Introduction to Copulas*.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
        return new{typeof(C),typeof(m)}(C,m)
    end    
end
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
  
function Distributions.cdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins}
    return Distributions.cdf(S.C,Distributions.cdf.(S.m,x))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, x::AbstractVector{T}) where {CT,TplMargins,T}
    Random.rand!(rng,S.C,x)
     x .= Distributions.quantile.(S.m,x)
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins},u) where {CT,TplMargins}
    sum(Distributions.logpdf(S.m[i],u[i]) for i in eachindex(u)) + Distributions.logpdf(S.C,clamp.(Distributions.cdf.(S.m,u),0,1))
end
function Distributions.fit(::Type{SklarDist{CT,TplMargins}},x) where {CT,TplMargins}
    # The first thing to do is to fit the marginals : 
    @assert length(TplMargins.parameters) == size(x,1)
    m = Tuple(Distributions.fit(TplMargins.parameters[i],x[i,:]) for i in axes(x,1))
    u = pseudos(x)
    C = Distributions.fit(CT,u)
    return SklarDist(C,m)
end