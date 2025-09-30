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
function (TG::Type{<:Generator})(args...;kwargs...)
    S = hasproperty(TG, :body) ? TG.body : TG
    T = S.name.wrapper 
    return T(args..., values(kwargs)...)
end
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

ρ(::IndependentGenerator)  = 0







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

struct FrailtyGenerator{TF}<:AbstractFrailtyGenerator
    F::TF
    function FrailtyGenerator(F::Distributions.ContinuousUnivariateDistribution)
        @assert Base.minimum(F) > 0
        return new{typeof(F)}(F)
    end
end
Distributions.params(G::FrailtyGenerator) = Distributions.params(G.F)
frailty(G::FrailtyGenerator) = G.F

# Add univaraite generator bindins: 
abstract type AbstractUnivariateGenerator <: Generator end
abstract type AbstractUnivariateFrailtyGenerator <: AbstractFrailtyGenerator end
const UnivariateGenerator = Union{AbstractUnivariateGenerator,AbstractUnivariateFrailtyGenerator}


"""
    WilliamsonGenerator{d, TX}
    i𝒲{TX}

Fields:
* `X::TX` -- a random variable that represents its Williamson d-transform

The type parameter `d::Int` is the dimension of the transformation. 

Constructor

    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)
    i𝒲(X::Distributions.UnivariateDistribution,d)
    WilliamsonGenerator(atoms::AbstractVector, weights::AbstractVector, d)
    i𝒲(atoms::AbstractVector, weights::AbstractVector, d)

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


Special case (finite-support discrete X)

- If `X isa Distributions.DiscreteUnivariateDistribution` and `support(X)` is finite, or if you pass directly atoms and weights to the constructor, the produced generator is piecewise-polynomial `ϕ(t) = ∑_j w_j · (1 − t/r_j)_+^(d−1)` matching the Williamson transform of a discrete radial law. It has specialized methods. 
- For infinite-support discrete distributions or when the support is not accessible as a finite
    iterable, the standard `WilliamsonGenerator` is constructed and will defer to
    `WilliamsonTransforms.jl`.

References: 
* [williamson1955multiply](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct WilliamsonGenerator{d, TX} <: Generator
    X::TX
    function WilliamsonGenerator(X, D::Val{d}) where d
        if X isa Distributions.DiscreteUnivariateDistribution
            # If X has finite, positive support, build an empirical generator
            sp = collect(Distributions.support(X))
            ws = Distributions.pdf.(X, sp)
            keep = ws .> 0
            return WilliamsonGenerator(sp[keep], ws[keep], D)
        end
        # else: fall back to a regular Williamson generator
        # check that X is indeed a positively supported random variable... 
        return new{d, typeof(X)}(X)
    end
    function WilliamsonGenerator(r::AbstractVector, w::AbstractVector, ::Val{d}) where d
        length(r) == length(w) || throw(ArgumentError("length(r) != length(w)"))
        !isempty(r) || throw(ArgumentError("no atoms given"))
        all(isfinite, r) && all(>=(0), r) || throw(ArgumentError("atoms must be positive and finite"))
        all(isfinite, w) && all(>(0), w) || throw(ArgumentError("weights must be positive and finite"))
        if !issorted(r)
            p = sortperm(r)
            r = r[p]; w = w[p]
        end
        # normalize
        X = Distributions.DiscreteNonParametric(r ./ r[end], w ./ sum(w); check_args=false)
        return new{d, typeof(X)}(X)
    end
end
const i𝒲 = WilliamsonGenerator
WilliamsonGenerator(X, d::Int) = WilliamsonGenerator(X, Val(d))
WilliamsonGenerator(r, w, d::Int) = WilliamsonGenerator(r, w, Val(d))
Distributions.params(G::WilliamsonGenerator) = (G.X,)
max_monotony(::WilliamsonGenerator{d, TX}) where {d, TX} = d
williamson_dist(G::WilliamsonGenerator{d, TX}, ::Val{d}) where {d, TX} = G.X # if its the right dim. 
ϕ(G::WilliamsonGenerator{d, TX}, t) where {d, TX} = WilliamsonTransforms.𝒲(G.X, Val{d}())(t)

# TODO: The following method for Kendall's tau is currently faulty and produces incorrect results.
# τ(G::WilliamsonGenerator) = 4*Distributions.expectation(Base.Fix1(ϕ, G), Copulas.williamson_dist(G, Val(2)))-1 # McNeil & Neshelova 2009
# Investigate the correct formula for Kendall's tau for WilliamsonGenerator. Check if the expectation is being computed with respect to the correct measure and if the implementation matches the reference (McNeil & Nešlehová 2009). Fix this method when the correct approach is established.
"""
    EmpiricalGenerator(u::AbstractMatrix)

Nonparametric Archimedean generator fit via inversion of the empirical Kendall distribution.

This function returns a `WilliamsonGenerator{TX}` whose underlying distribution `TX` is a `Distributions.DiscreteNonParametric`, rather than a separate struct.
The returned object still implements all optimized methods (ϕ, derivatives, inverses) via specialized dispatch on `WilliamsonGenerator{<:DiscreteNonParametric}`.

Usage

    G = EmpiricalGenerator(u)

where `u::AbstractMatrix` is a `d×n` matrix of observations (already on copula or pseudo scale).

Notes
* The recovered discrete radial support is rescaled so its largest atom equals 1 (scale is not identifiable).
* We keep the old documentation entry point for backward compatibility; existing code that
  relied on the `EmpiricalGenerator` type should instead treat the result as a `Generator`.

References
* [mcneil2009multivariate](@cite)
* [williamson1956](@cite)
* [genest2011a](@cite) Genest, Neslehova and Ziegel (2011), Inference in Multivariate Archimedean Copula Models
"""
function EmpiricalGenerator(u::AbstractMatrix)
    d = size(u, 1)
    W = _kendall_sample(u)
    kw = StatsBase.proportionmap(W)
    x = collect(keys(kw))
    N = length(x)
    N == 1 && return ClaytonGenerator(-1/(d-1))
    sort!(x; rev=true)
    w = [kw[xi] for xi in x]
    r = zero(x)
    r[end] = 1
    r[end-1] = 1 - clamp(x[N-1] / w[N], 0, 1)^(1/(d-1))
    for k in (N-2):-1:1
        gk = function(y)
            s = 0.0
            @inbounds for j in (k+1):N
                z = 1.0 - y / r[j]
                if z > 0.0
                    s += w[j] * z^(d-1)
                end
            end
            return s
        end
        eps = 1e-14
        a, b = 0.0, max(r[k+1] - eps, 0.0)
        ga, gb = gk(a), gk(b)
        if !(ga + 1e-12 >= x[k] >= gb - 1e-12)
            a, b = 0.0, r[k+1]
        end
        r[k] = Roots.find_zero(y -> gk(y) - x[k], (a, b); verbose=false)
        r[k] = clamp(r[k], 0.0, r[k+1] - eps)
    end
    return WilliamsonGenerator(r, w, Val(d))
end

# Optimized methods for discrete nonparametric Williamson generators (covers EmpiricalGenerator)
function ϕ(G::WilliamsonGenerator{d, TX}, t::Real) where {d, TX<:Distributions.DiscreteNonParametric}
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t))
    t <= 0 && return one(Tt)
    t >= r[end] && return zero(Tt)
    S = zero(Tt)
    @inbounds for j in lastindex(r):-1:firstindex(r)
        rⱼ = r[j]; wⱼ = w[j]
        t >= rⱼ && break
        S += wⱼ * (1 - t / rⱼ)^(d - 1)
    end
    return S
end

function ϕ⁽¹⁾(G::WilliamsonGenerator{d, TX}, t::Real) where {d, TX<:Distributions.DiscreteNonParametric}
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t))
    t >= r[end] && return zero(Tt)
    S = zero(Tt)
    @inbounds for j in lastindex(r):-1:firstindex(r)
        rⱼ = r[j]; wⱼ = w[j]
        t ≥ rⱼ && break
        zpow = d==2 ? one(t) : (1 - t / rⱼ)^(d-2)
        S += wⱼ * zpow / rⱼ
    end
    return - (d-1) * S
end

function ϕ⁽ᵏ⁾(G::WilliamsonGenerator{d, TX}, ::Val{k}, t::Real) where {d, k, TX<:Distributions.DiscreteNonParametric}
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t))
    (k >= d || t >= r[end]) && return zero(Tt)
    k == 0 && return ϕ(G, t)
    k == 1 && return ϕ⁽¹⁾(G, t)
    S = zero(Tt)
    @inbounds for j in lastindex(r):-1:firstindex(r)
        rⱼ = r[j]; wⱼ = w[j]
        t ≥ rⱼ && break
        zpow = (d == k+1) ? one(t) : (1 - t / rⱼ)^(d - 1 - k)
        S += wⱼ * zpow / rⱼ^k
    end
    return S * (isodd(k) ? -1 : 1) * Base.factorial(d - 1) / Base.factorial(d - 1 - k)
end

function ϕ⁻¹(G::WilliamsonGenerator{d, TX}, x::Real) where {d, TX<:Distributions.DiscreteNonParametric}
    r = Distributions.support(G.X)
    Tx = promote_type(eltype(r), typeof(x))
    x >= 1 && return zero(Tx)
    x <= 0 && return Tx(r[end])
    for k in eachindex(r)
        ϕ_rk = ϕ(G, r[k])
        if x > ϕ_rk
            if x < ϕ(G, prevfloat(r[k]))
                return Tx(prevfloat(r[k]))
            end
            a = (k==1 ? 0 : r[k-1]); b = r[k]
            return Tx(Roots.find_zero(t -> ϕ(G, t) - x, (a, b); bisection=true))
        end
    end
    return Tx(r[end])
end

function ϕ⁽ᵏ⁾⁻¹(G::WilliamsonGenerator{d, TX}, ::Val{p}, y; start_at=nothing) where {d, p, TX<:Distributions.DiscreteNonParametric}
    r = Distributions.support(G.X)
    Ty = promote_type(eltype(r), typeof(y))
    p == 0 && return ϕ⁻¹(G, y)
    sign = iseven(p) ? 1 : -1
    s_y = sign*y
    s_y <= 0 && return Ty(r[end])
    s_y >= sign*ϕ⁽ᵏ⁾(G, Val{p}(), 0) && return Ty(0)
    for k in eachindex(r)
        ϕp_rk = sign * ϕ⁽ᵏ⁾(G, Val{p}(), r[k])
        if s_y > ϕp_rk
            if s_y < sign * ϕ⁽ᵏ⁾(G, Val{p}(), prevfloat(r[k]))
                return Ty(prevfloat(r[k]))
            end
            a = (k==1 ? 0 : r[k-1]); b = r[k]
            return Ty(Roots.find_zero(t -> ϕ⁽ᵏ⁾(G, Val{p}(), t) - y, (a, b); bisection=true))
        end
    end
    return Ty(r[end])
end






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
Distributions.params(G::TiltedGenerator) = (Distributions.params(G.G)..., sJ = G.sJ)