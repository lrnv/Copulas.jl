"""
    FrailtyGenerator<:AbstractFrailtyGenerator<:Generator

methods: 
    - frailty(::FrailtyGenerator) gives the frailty 
    - ϕ and the rest of generators are automatically defined from the frailty. 

Constructor

    FrailtyGenerator(D)

A Frailty generator can be defined by a positive random variable that happens to have a `mgf()` 
function to compute its moment generating function. The generator is simply: 

```math
\\phi(t) = mgf(frailty(G), -t)
```

https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/2009-08-16_hofert.pdf

References:
* [hofert2009](@cite) M. Hoffert (2009). Efficiently sampling Archimedean copulas
"""
FrailtyGenerator

abstract type AbstractFrailtyGenerator<:Generator end
frailty(::AbstractFrailtyGenerator) = throw("This generator was not defined as it should, you should provide its frailty")
max_monotony(::AbstractFrailtyGenerator) = Inf
ϕ(G::AbstractFrailtyGenerator, t) = Distributions.mgf(frailty(G), -t)
williamson_dist(G::AbstractFrailtyGenerator, ::Val{d}) where d = WilliamsonFromFrailty(frailty(G), Val{d}())
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d, GT}, x::AbstractVector{T}) where {T<:Real, d, GT<:AbstractFrailtyGenerator}
    F = frailty(C.G)
    Random.randexp!(rng, x)
    f = rand(rng, F)
    x .= ϕ.(C.G, x ./ f)
    return x
end

struct FrailtyGenerator{TF}<:AbstractFrailtyGenerator
    F::TF
    function FrailtyGenerator(F::Distributions.ContinuousUnivariateDistribution)
        @assert Base.minimum(F) > 0
        return new{typeof(F)}(F)
    end
end
Distributions.params(G::FrailtyGenerator) = Distributions.params(G.F)
frailty(G::FrailtyGenerator) = G.F