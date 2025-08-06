"""
    FrailtyGenerator{T}

Fields:
  - D::Distributions.Distribution - parameter, shoudl be the distribution of a positive random variable.

Constructor

    FrailtyGenerator(D)

A Frailty generator can be defined by a positive random variable that happens to have a `mgf()` function to compute its moment generating function. The generator is simply: 

```math
\\phi(t) = mgf(D, -t)
```


https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/2009-08-16_hofert.pdf

References:
* [hofert2009](@cite) M. Hoffert (2009). Efficiently sampling Archimedean copulas
"""
struct FrailtyGenerator{TD} <: Generator
    D::TD
    function FrailtyGenerator(D)
        @assert isa(D, Distributions.UnivariateDistribution)
        @assert minimum(D) >= 0
        return new{typeof(D)}(D)
    end
end

max_monotony(::FrailtyGenerator) = Inf
ϕ(G::FrailtyGenerator, t) = Distributions.mgf(G.D, -t)
williamson_dist(G::FrailtyGenerator, ::Val{d}) where d = WilliamsonFromFrailty(G.D, Val{d}())

# ϕ⁻¹(G::FrailtyGenerator, t) = 
# ϕ⁽¹⁾(G::FrailtyGenerator, t) = 
# ϕ⁻¹⁽¹⁾(G::FrailtyGenerator, t) = 
# ϕ⁽ᵏ⁾(G::FrailtyGenerator, ::Val{k}, t) where k = 
# ϕ⁽ᵏ⁾⁻¹(G::FrailtyGenerator, ::Val{k}, t; start_at=t) where k =   
# τ(G::FrailtyGenerator) =
# τ⁻¹(::Type{T},τ) where T<:FrailtyGenerator = 


function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d, FrailtyGenerator{TD}}, A::DenseMatrix{<:Real}) where {d, TD}
    v = rand(rng, C.G.D, size(A, 2))
    rand!(rng, A)
    A .= .- log.(A) ./ v'
    return ϕ.(C.G, A)
end