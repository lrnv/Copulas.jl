"""
    Generator

Abstract type. Implements the API for archimedean generators.

An Archimedean generator is simply a function
``\\phi :\\mathbb R_+ \\to [0,1]`` such that ``\\phi(0) = 1`` and ``\\phi(+\\infty) = 0``.

To generate an archimedean copula in dimension ``d``, the function also needs to be ``d``-monotone, that is :

- ``\\phi`` is ``d-2`` times derivable.
- ``(-1)^k \\phi^{(k)} \\ge 0 \\;\\forall k \\in \\{1,..,d-2\\},`` and if ``(-1)^{d-2}\\phi^{(d-2)}`` is a non-increasing and convex function.

The access to the function ``\\phi`` itself is done through the interface:

    ϕ(G::Generator, t)

We do not check algorithmically that the proposed generators are d-monotonous. Instead, it is up to the person implementing the generator to tell the interface how big can ``d`` be through the function

    max_monotony(G::MyGenerator) = # some integer, the maximum d so that the generator is d-monotonous.


More methods can be implemented for performance, althouhg there are implement defaults in the package :

* `ϕ⁻¹( G::Generator, x)` gives the inverse function of the generator.
* `ϕ⁽¹⁾(G::Generator, t)` gives the first derivative of the generator
* `ϕ⁽ᵏ⁾(G::Generator, ::Val{k}, t) where k` gives the kth derivative of the generator
* `ϕ⁻¹⁽¹⁾(G::Generator, t)` gives the first derivative of the inverse generator.
* `williamson_dist(G::Generator, ::Val{d}) where d` gives the Wiliamson d-transform of the generator, see [WilliamsonTransforms.jl](https://github.com/lrnv/WilliamsonTransforms.jl).

References:
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
"""
abstract type Generator end
Base.broadcastable(x::Generator) = Ref(x)
max_monotony(G::Generator) = throw("This generator does not have a defined max monotony. You need to implement `max_monotony(G)`.")
ϕ(   G::Generator, t) = throw("This generator has not been defined correctly, the function `ϕ(G,t)` is not defined.")
ϕ(G::Generator) = Base.Fix1(ϕ,G)
ϕ⁻¹( G::Generator, x) = Roots.find_zero(t -> ϕ(G,t) - x, (0.0, Inf))
ϕ⁽¹⁾(G::Generator, t) = ForwardDiff.derivative(x -> ϕ(G,x), t)
ϕ⁻¹⁽¹⁾(G::Generator, t) = ForwardDiff.derivative(x -> ϕ⁻¹(G, x), t)
ϕ⁽ᵏ⁾(G::Generator, ::Val{k}, t) where k = WilliamsonTransforms.taylor(ϕ(G), t, Val{k}())[end] * factorial(k)
ϕ⁽ᵏ⁾⁻¹(G::Generator, ::Val{k}, t; start_at=t) where {k} = Roots.find_zero(x -> ϕ⁽ᵏ⁾(G, Val{k}(), x) - t, start_at)
williamson_dist(G::Generator, ::Val{d}) where d = WilliamsonTransforms.𝒲₋₁(ϕ(G), Val{d}())


# TODO: Move the \phi^(1) to defer to \phi^(k=1), and implement \phi(k=1) in generators instead of \phi^(1)
# That would help a lot the performance of some routines. 
# But its a bit hard to do as it modifies a lot of files.


# τ(G::Generator) = @error("This generator has no kendall tau implemented.")
# ρ(G::Generator) = @error ("This generator has no Spearman rho implemented.")
# τ⁻¹(G::Generator, τ_val) = @error("This generator has no inverse kendall tau implemented.")
# ρ⁻¹(G::Generator, ρ_val) = @error ("This generator has no inverse Spearman rho implemented.")

struct IndependentGenerator <: Generator end 
struct MGenerator <: Generator end
struct WGenerator <: Generator end

τ(::IndependentGenerator)  = 0
τ(::MGenerator)  = 1
τ(::WGenerator)  = -1







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







"""
    WilliamsonGenerator{TX}
    i𝒲{TX}

Fields:
* `X::TX` -- a random variable that represents its Williamson d-transform
* `d::Int` -- the dimension of the transformation. 

Constructor

    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)
    i𝒲(X::Distributions.UnivariateDistribution,d)

The `WilliamsonGenerator` (alias `i𝒲`) allows to construct a d-monotonous archimedean generator from a positive random variable `X::Distributions.UnivariateDistribution`. The transformation, which is called the inverse Williamson transformation, is implemented in [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl). 

For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer ``d\\ge 2``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

This function has several properties: 
- We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

These properties makes this function what is called a *d-monotone archimedean generator*, able to generate *archimedean copulas* in dimensions up to ``d``. Our implementation provides this through the `Generator` interface: the function ``\\phi`` can be accessed by 

    G = WilliamsonGenerator(X, d)
    ϕ(G,t)

Note that you'll always have:

    max_monotony(WilliamsonGenerator(X,d)) === d

References: 
* [williamson1955multiply](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct WilliamsonGenerator{TX} <: Generator
    X::TX
    d::Int
    function WilliamsonGenerator(X,transform_dimension)
        # check that X is indeed a positively supported random variable... 
        return new{typeof(X)}(X,transform_dimension)
    end
end
const i𝒲 = WilliamsonGenerator
Distributions.params(G::WilliamsonGenerator) = (G.X, G.d)
function Base.show(io::IO, G::WilliamsonGenerator)
    print(io, "i𝒲($(G.X), $(G.d))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, TG}) where {d, TG<:WilliamsonGenerator}
    print(io, "ArchimedeanCopula($(length(C)), i𝒲($(C.G.X), $(C.G.d)))")
end
max_monotony(G::WilliamsonGenerator) = G.d
function williamson_dist(G::WilliamsonGenerator, ::Val{d}) where d
    if d == G.d 
        return G.X
    end
    # what about d < G.d ? Mayeb we can do some frailty stuff ? 
    return WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t), Val{d}())
end
ϕ(G::WilliamsonGenerator, t) = WilliamsonTransforms.𝒲(G.X, Val(G.d))(t)

# McNeil & Neshelova 2009
τ(G::WilliamsonGenerator) = 4*Distributions.expectation(x -> Copulas.ϕ(G,x), Copulas.williamson_dist(G, Val(2)))-1







"""
    TiltedGenerator(G, p, sJ) <: Generator

Archimedean generator tilted by conditioning on `p` components fixed at values
with cumulative generator sum `sJ = ∑ ϕ⁻¹(u_j)`. It defines

    ϕ_tilt(t) = ϕ^{(p)}(sJ + t) / ϕ^{(p)}(sJ)

and higher derivatives accordingly:

    ϕ_tilt^{(k)}(t) = ϕ^{(k+p)}(sJ + t) / ϕ^{(p)}(sJ)

which yields the conditional copula within the Archimedean family for the
remaining d-p variables.
You will get a TiltedGenerator if you condition() an archimedean copula.
"""
struct TiltedGenerator{TG, T, p} <: Generator
    G::TG
    sJ::T
    den::T
    function TiltedGenerator(G::Generator, ::Val{p}, sJ::T) where {p,T<:Real}
        den = ϕ⁽ᵏ⁾(G, Val{p}(), sJ)
        return new{typeof(G), T, p}(G, sJ, den)
    end
end
max_monotony(G::TiltedGenerator{TG, T, p}) where {TG, T, p} = max(0, max_monotony(G.G) - p)
ϕ(G::TiltedGenerator{TG, T, p}, t::Real) where {TG, T, p} = ϕ⁽ᵏ⁾(G.G, Val{p}(), G.sJ + t) / G.den
ϕ⁻¹(G::TiltedGenerator{TG, T, p}, x::Real) where {TG, T, p} = ϕ⁽ᵏ⁾⁻¹(G.G, Val{p}(), x * G.den; start_at = G.sJ) - G.sJ
ϕ⁽ᵏ⁾(G::TiltedGenerator{TG, T, p}, ::Val{k}, t::Real) where {TG, T, p, k} = ϕ⁽ᵏ⁾(G.G, Val{k + p}(), G.sJ + t) / G.den
ϕ⁽ᵏ⁾⁻¹(G::TiltedGenerator{TG, T, p}, ::Val{k}, y::Real; start_at = G.sJ) where {TG, T, p, k} = ϕ⁽ᵏ⁾⁻¹(G.G, Val{k + p}(), y * G.den; start_at = start_at) - G.sJ
ϕ⁽¹⁾(G::TiltedGenerator{TG, T, p}, t) where {TG, T, p} = ϕ⁽ᵏ⁾(G, Val{1}(), t)