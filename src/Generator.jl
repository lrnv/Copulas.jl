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
* `ϕ⁽ᵏ⁾(G::Generator, k::Int, t)` gives the kth derivative of the generator
* `ϕ⁻¹⁽¹⁾(G::Generator, t)` gives the first derivative of the inverse generator.
* `𝒲₋₁(G::Generator, d::Real)` gives the inverse Williamson transform of the generator as a positive univariate distribution. Positive non-integer orders use an exact beta reduction from `ceil(Int, d)`.

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
function ϕ⁽ᵏ⁾(G::Generator, k::Int, t)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    return _mul_factorial(taylor(ϕ(G), t, k)[end], k)
end
function ϕ⁽ᵏ⁾⁻¹(G::Generator, k::Int, t; start_at=t)
    f(x) = ϕ⁽ᵏ⁾(G, k, x) - t
    T = typeof(float(t))
    lo, hi = eps(T), one(T)
    flo, fhi = f(lo), f(hi)
    iszero(flo) && return lo
    iszero(fhi) && return hi

    for _ in 1:64
        signbit(flo) != signbit(fhi) &&
            return Roots.find_zero(f, (lo, hi), Roots.Bisection())
        hi *= 2
        fhi = f(hi)
        iszero(fhi) && return hi
    end
    throw(ArgumentError("Could not bracket the inverse generator derivative"))
end



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
    𝒲₋₁(G::Generator, d::Real)

Computes the inverse Williamson transform of the monotone Archimedean generator
`G` at a positive real order `d`.

For an integer order, the generic implementation uses the classical inversion
formula below, while more specific generator families may provide an exact or
faster radial distribution. For non-integer `d`, it first inverts at
`n = ceil(Int, d)` and returns the law of `Rₙ * B`, where
`B ~ Beta(d, n-d)` is independent of `Rₙ = 𝒲₋₁(G, n)`. Consequently,
`ceil(d) <= max_monotony(G)` is required. Integer-valued orders retain the
specialized integer dispatch path. If `G = 𝒲(X, source_order)` retains its
source radial, every `d <= source_order` is instead reduced directly from `X`;
the ceiling condition is then unnecessary.

A ``d``-monotone archimedean generator is a function ``\\phi`` on ``\\mathbb R_+`` that has these three properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

For such a function ``\\phi``, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``X``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\\phi)(x) = 1 - \\frac{(-x)^{d-1} \\phi_+^{(d-1)}(x)}{k!} - \\sum_{k=0}^{d-2} \\frac{(-x)^k \\phi^{(k)}(x)}{k!}
```

We return this cumulative distribution function in the form of the corresponding random variable `<:Distributions.ContinuousUnivariateDistribution` from `Distributions.jl`. You may then compute : 
    - The cdf via `Distributions.cdf`
    - The pdf via `Distributions.pdf` and the logpdf via `Distributions.logpdf`
    - Samples from the distribution via `rand(X,n)`

References: 
    - Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    - McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct 𝒲₋₁{TG, TO<:Integer} <: Distributions.ContinuousUnivariateDistribution
    # Woul dprobably be much more efficient if it took the generator and not the function itself. 
    G::TG
    order::TO
    function 𝒲₋₁(G::Generator, d::Integer)
        @assert max_monotony(G) ≥ d
        d ≥ 1 || throw(ArgumentError("the Williamson inverse order must be at least 1"))
        return new{typeof(G), typeof(d)}(G, d)
    end
end

function 𝒲₋₁(G::Generator, d::Real)
    isfinite(d) && d > 0 || throw(ArgumentError("the Williamson order must be finite and positive"))
    n = ceil(Int, d)
    n <= max_monotony(G) || throw(ArgumentError(
        "cannot invert a generator of maximal monotonicity $(max_monotony(G)) at order $d",
    ))
    isinteger(d) && return 𝒲₋₁(G, n)
    return WilliamsonBetaProduct(𝒲₋₁(G, n), Distributions.Beta(d, n - d))
end
function Distributions.cdf(dist::𝒲₋₁, x::Real)
    x ≤ 0 && return zero(x)
    rez, scaled_power = zero(x), one(x)
    @inbounds for k in 1:dist.order
        cₖ = if k == 1
            ϕ(dist.G, x)
        elseif k == 2
            ϕ⁽¹⁾(dist.G, x)
        else
            ϕ⁽ᵏ⁾(dist.G, k-1, x)
        end
        rez += scaled_power * cₖ
        scaled_power *= -x / k
    end
    F = 1 - rez
    # Guard against tiny numerical excursions
    return isnan(F) ? one(x) : clamp(F, zero(x), one(x))
end
function Distributions.pdf(dist::𝒲₋₁, x::Real)
    x ≤ 0 && return zero(x)
    isinf(x) && return zero(float(x))
    # Differentiating the inverse-Williamson CDF makes all intermediate
    # terms telescope: f_R(x) = (-1)^d x^(d-1) ϕ^(d)(x) / (d-1)!.
    scale = one(float(x))
    @inbounds for k in 1:(dist.order - 1)
        scale *= x / k
    end
    density = (isodd(dist.order) ? -scale : scale) * ϕ⁽ᵏ⁾(dist.G, dist.order, x)
    return max(zero(density), density)
end
Distributions.logpdf(dist::𝒲₋₁, x) = log(Distributions.pdf(dist, x))
_quantile(dist::𝒲₋₁, p) = Roots.find_zero(x -> (Distributions.cdf(dist, x) - p), (0.0, Inf))
Distributions.rand(rng::Distributions.AbstractRNG, dist::𝒲₋₁) = _quantile(dist, rand(rng))
Base.minimum(::𝒲₋₁) = 0.0
Base.maximum(::𝒲₋₁) = Inf
function Distributions.quantile(dist::𝒲₋₁, p::Real)
    @assert 0 <= p <= 1
    return _quantile(dist, p)
end

# Radial law of a lower-order margin. If ψ = W_D(F_R), Dirichlet
# aggregation gives W_d⁻¹(ψ) = Law(RB), B ~ Beta(d, D-d), independently.
struct WilliamsonBetaProduct{TX, TB} <: Distributions.ContinuousUnivariateDistribution
    X::TX
    B::TB
end

function WilliamsonBetaProduct(X::WilliamsonFromFrailty, B::Distributions.Beta)
    target_order, order_gap = Distributions.params(B)
    target_order + order_gap == X.order ||
        return WilliamsonBetaProduct{typeof(X),typeof(B)}(X, B)
    return WilliamsonFromFrailty(X.frailty_dist, target_order)
end

function WilliamsonBetaProduct(X::WilliamsonBetaProduct, B::Distributions.Beta)
    inner_target, inner_gap = Distributions.params(X.B)
    outer_target, outer_gap = Distributions.params(B)
    if outer_target + outer_gap == inner_target
        source_order = inner_target + inner_gap
        merged_beta = Distributions.Beta(outer_target, source_order - outer_target)
        return WilliamsonBetaProduct(X.X, merged_beta)
    end
    return WilliamsonBetaProduct{typeof(X), typeof(B)}(X, B)
end

function Distributions.cdf(dist::WilliamsonBetaProduct, x::Real)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(dist.X) do r
        r <= x ? one(float(x)) : Distributions.cdf(dist.B, x / r)
    end
end

function Distributions.pdf(dist::WilliamsonBetaProduct, x::Real)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(dist.X) do r
        r <= x ? zero(float(x)) : Distributions.pdf(dist.B, x / r) / r
    end
end

# For continuous radials, conditioning on B integrates over its bounded support
# and reuses the radial distribution's specialized cdf/pdf implementations.
function Distributions.cdf(
    dist::WilliamsonBetaProduct{<:Distributions.ContinuousUnivariateDistribution},
    x::Real,
)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(b -> Distributions.cdf(dist.X, x / b), dist.B)
end

function Distributions.pdf(
    dist::WilliamsonBetaProduct{<:Distributions.ContinuousUnivariateDistribution},
    x::Real,
)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(
        b -> iszero(b) ? zero(float(x)) : Distributions.pdf(dist.X, x / b) / b,
        dist.B,
    )
end

Distributions.logpdf(dist::WilliamsonBetaProduct, x::Real) = log(Distributions.pdf(dist, x))
Distributions.rand(rng::Distributions.AbstractRNG, dist::WilliamsonBetaProduct) =
    rand(rng, dist.X) * rand(rng, dist.B)
Base.minimum(dist::WilliamsonBetaProduct) = zero(float(Base.minimum(dist.X)))
Base.maximum(dist::WilliamsonBetaProduct) = Base.maximum(dist.X)

function _positive_distribution_quantile(dist, p::Real)
    lo = float(Base.minimum(dist))
    hi = float(Base.maximum(dist))
    if !isfinite(hi)
        hi = max(one(lo), lo + one(lo))
        while Distributions.cdf(dist, hi) < p
            hi *= 2
            isfinite(hi) || return hi
        end
    end
    return Roots.find_zero(
        x -> Distributions.cdf(dist, x) - p,
        (lo, hi),
        Roots.Bisection(),
    )
end

function Distributions.quantile(dist::WilliamsonBetaProduct, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    iszero(p) && return Base.minimum(dist)
    isone(p) && return Base.maximum(dist)
    return _positive_distribution_quantile(dist, p)
end




"""
    𝒲{TX, TO} (alias WilliamsonGenerator{TX, TO})

Fields:
* `X::TX` -- a random variable that represents its Williamson d-transform
* `order::TO` -- the order of the Williamson transform

The type parameter `TO` is the numeric type of the order, not its value.

Constructor

    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)
    𝒲(X::Distributions.UnivariateDistribution,d)
    WilliamsonGenerator(atoms::AbstractVector, weights::AbstractVector, d)
    𝒲(atoms::AbstractVector, weights::AbstractVector, d)

The `𝒲` type (also available as `WilliamsonGenerator`) constructs a d-monotonous archimedean generator from a positive random variable `X::Distributions.UnivariateDistribution`. The transformation is implemented fully generically in the package.

For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and a positive real order ``d``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

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

    max_monotony(WilliamsonGenerator(X,d)) == d


Special case (finite-support discrete X)

- If `X isa Distributions.DiscreteUnivariateDistribution` and `support(X)` is finite, or if you pass directly atoms and weights to the constructor, the produced generator is piecewise-polynomial `ϕ(t) = ∑_j w_j · (1 − t/r_j)_+^(d−1)` matching the Williamson transform of a discrete radial law. It has specialized methods. 
- For infinite-support discrete distributions or when the support is not accessible as a finite
    iterable, the standard `WilliamsonGenerator` is constructed.

References: 
* [williamson1956](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct 𝒲{TX, TO<:Real} <: Generator
    X::TX
    order::TO
    function 𝒲(X, d::Real)
        isfinite(d) && d > 0 || throw(ArgumentError("the Williamson order must be finite and positive"))
        if X isa Distributions.DiscreteNonParametric
            # If X has finite, positive support, build an empirical generator
            sp = collect(Distributions.support(X))
            ws = Distributions.pdf.(X, sp)
            keep = ws .> 0
            return 𝒲(sp[keep], ws[keep], d)
        end
        # else: fall back to a regular Williamson generator
        # check that X is indeed a positively supported random variable... 
        return new{typeof(X), typeof(d)}(X, d)
    end
    function 𝒲(r::AbstractVector, w::AbstractVector, d::Real)
        isfinite(d) && d > 0 || throw(ArgumentError("the Williamson order must be finite and positive"))
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
        return new{typeof(X), typeof(d)}(X, d)
    end
end
const WilliamsonGenerator = 𝒲
@doc (@doc 𝒲) WilliamsonGenerator
Distributions.params(G::𝒲) = (G.X,)
max_monotony(G::𝒲) = G.order
"""
Generic fallback for ϕ on WilliamsonGenerator (non-discrete-nonparametric TX).
Specializations for `TX<:DiscreteNonParametric` are provided below.
"""
function ϕ(G::𝒲, t)
    t <= 0 && return one(t)
    return Distributions.expectation(y -> (y > t) ? (1 - t / y)^(G.order - 1) : zero(t), G.X)
end

function ϕ⁽ᵏ⁾(G::𝒲, k::Int, t)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    k == 0 && return ϕ(G, t)
    t < 0 && return zero(float(t))
    k < G.order || return invoke(ϕ⁽ᵏ⁾, Tuple{Generator, Int, Any}, G, k, t)

    coefficient = _falling_factorial(G.order - 1, k)
    value = Distributions.expectation(G.X) do y
        y > t ? (1 - t / y)^(G.order - 1 - k) / y^k : zero(t + y + G.order)
    end
    return (isodd(k) ? -coefficient : coefficient) * value
end
ϕ⁽¹⁾(G::𝒲, t) = ϕ⁽ᵏ⁾(G, 1, t)
function ϕ(G::𝒲, x::TaylorSeries.Taylor1{TF}) where {TF}
    x <= 0 && return one(x) - Distributions.cdf(G.X,0)
    x₀ = x.coeffs[1]
    p = length(x.coeffs)
    rez = zeros(TF,p)
    for i in 1:p
        xᵢ = TaylorSeries.Taylor1(x.coeffs[1:i])
        fᵢ(y) = y ≤ x₀ ? zero(y) : ((1 - xᵢ/y)^(G.order-1)).coeffs[i]
        rez[i] = Distributions.expectation(fᵢ, G.X)
    end
    return TaylorSeries.Taylor1(rez)
end

# Exact inverse paths when the forward transform retains its radial law.
function _williamson_inverse_preserved(G::𝒲, d::Real)
    isfinite(d) && d > 0 || throw(ArgumentError("the Williamson order must be finite and positive"))
    d == G.order && return G.X
    d < G.order && return WilliamsonBetaProduct(G.X, Distributions.Beta(d, G.order - d))
    throw(ArgumentError("cannot invert a Williamson transform above its source order $(G.order)"))
end
𝒲₋₁(G::𝒲, d::Integer) = _williamson_inverse_preserved(G, d)
𝒲₋₁(G::𝒲, d::Real) = _williamson_inverse_preserved(G, d)
function 𝒲(X::𝒲₋₁, d::Real)
    d == X.order && return X.G
    return invoke(𝒲, Tuple{Any, Real}, X, d)
end
function 𝒲(X::WilliamsonBetaProduct, d::Real)
    target_order, order_gap = Distributions.params(X.B)
    d == target_order && return 𝒲(X.X, target_order + order_gap)
    return invoke(𝒲, Tuple{Any, Real}, X, d)
end


# Optimized methods for discrete nonparametric Williamson generators (covers EmpiricalGenerator)
function ϕ(G::𝒲{<:Distributions.DiscreteNonParametric}, t)
    d = G.order
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t), typeof(d))
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

function ϕ⁽¹⁾(G::𝒲{<:Distributions.DiscreteNonParametric}, t)
    d = G.order
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t), typeof(d))
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

function ϕ⁽ᵏ⁾(G::𝒲{<:Distributions.DiscreteNonParametric}, k::Int, t)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    d = G.order
    r = Distributions.support(G.X)
    w = Distributions.probs(G.X)
    Tt = promote_type(eltype(r), typeof(t), typeof(d))
    t >= r[end] && return zero(Tt)
    k == 0 && return ϕ(G, t)
    k == 1 && return ϕ⁽¹⁾(G, t)
    S = zero(Tt)
    @inbounds for j in lastindex(r):-1:firstindex(r)
        rⱼ = r[j]; wⱼ = w[j]
        t ≥ rⱼ && break
        zpow = (d == k+1) ? one(t) : (1 - t / rⱼ)^(d - 1 - k)
        S += wⱼ * zpow / rⱼ^k
    end
    coefficient = _falling_factorial(Tt(d - 1), k)
    return S * (isodd(k) ? -1 : 1) * coefficient
end

function ϕ⁻¹(G::𝒲{<:Distributions.DiscreteNonParametric}, x)
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

function ϕ⁽ᵏ⁾⁻¹(G::𝒲{<:Distributions.DiscreteNonParametric}, p::Int, y; start_at=nothing)
    r = Distributions.support(G.X)
    Ty = promote_type(eltype(r), typeof(y))
    p == 0 && return ϕ⁻¹(G, y)
    sign = iseven(p) ? 1 : -1
    s_y = sign*y
    s_y <= 0 && return Ty(r[end])
    s_y >= sign*ϕ⁽ᵏ⁾(G, p, 0) && return Ty(0)
    for k in eachindex(r)
        ϕp_rk = sign * ϕ⁽ᵏ⁾(G, p, r[k])
        if s_y > ϕp_rk
            if s_y < sign * ϕ⁽ᵏ⁾(G, p, prevfloat(r[k]))
                return Ty(prevfloat(r[k]))
            end
            a = (k==1 ? 0 : r[k-1]); b = r[k]
            return Ty(Roots.find_zero(t -> ϕ⁽ᵏ⁾(G, p, t) - y, (a, b); bisection=true))
        end
    end
    return Ty(r[end])
end






"""
    EmpiricalGenerator(u::AbstractMatrix)

Nonparametric Archimedean generator fit via inversion of the empirical Kendall distribution.

This function returns a `WilliamsonGenerator{TX, TO}` whose underlying distribution `TX` is a `Distributions.DiscreteNonParametric`, rather than a separate struct.
The returned object still implements all optimized methods (ϕ, derivatives, inverses) via specialized dispatch on `WilliamsonGenerator{<:DiscreteNonParametric}`.

Usage

    G = EmpiricalGenerator(u)

where `u::AbstractMatrix` is a `d×n` matrix of observations (already on copula or pseudo scale).

Notes
* The recovered discrete radial support is rescaled so its largest atom equals 1 (scale is not identifiable).
* We keep the old documentation entry point for backward compatibility; existing code that
  relied on the `EmpiricalGenerator` type should instead treat the result as a `Generator`.

References
* [mcneil2009](@cite)
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
        # Ensure a valid bracket: gk is nonincreasing in y, target is x[k]
        # Expand upper bound slightly if needed to include the target
        if !(ga + 1e-12 >= x[k] >= gb - 1e-12)
            # Try with full [0, r[k+1]] first
            a, b = 0.0, r[k+1]
            ga, gb = gk(a), gk(b)
        end
        if !(ga >= x[k] >= gb)
            # As a last resort, project x[k] into [gb, ga]
            xk = clamp(x[k], gb, ga)
            r[k] = Roots.find_zero(y -> gk(y) - xk, (a, b); bisection=true)
        else
            r[k] = Roots.find_zero(y -> gk(y) - x[k], (a, b); bisection=true)
        end
        r[k] = clamp(r[k], 0.0, r[k+1] - eps)
    end
    return 𝒲(r, w, d)
end





"""
    TiltedGenerator(G, p, sJ)

Archimedean generator tilted by conditioning on `p` components fixed at values
with cumulative generator sum `sJ = ∑ ϕ⁻¹(u_j)`. It defines

    ϕ_tilt(t) = ϕ^{(p)}(sJ + t) / ϕ^{(p)}(sJ)

and higher derivatives accordingly:

    ϕ_tilt^{(k)}(t) = ϕ^{(k+p)}(sJ + t) / ϕ^{(p)}(sJ)

which yields the conditional copula within the Archimedean family for the
remaining d-p variables.
You will get a TiltedGenerator if you condition() an archimedean copula.
"""
struct TiltedGenerator{TG, T} <: Generator
    G::TG
    p::Int
    sJ::T
    den::T
    function TiltedGenerator(G::Generator, p::Int, sJ::T) where {T<:Real}
        den = ϕ⁽ᵏ⁾(G, p, sJ)
        return new{typeof(G), T}(G, p, sJ, den)
    end
end
max_monotony(G::TiltedGenerator{TG, T}) where {TG, T} = max(0, max_monotony(G.G) - G.p)
ϕ(G::TiltedGenerator{TG, T}, t) where {TG, T} = ϕ⁽ᵏ⁾(G.G, G.p, G.sJ + t) / G.den
ϕ⁻¹(G::TiltedGenerator{TG, T}, x) where {TG, T} = ϕ⁽ᵏ⁾⁻¹(G.G, G.p, x * G.den; start_at = G.sJ) - G.sJ
ϕ⁽ᵏ⁾(G::TiltedGenerator{TG, T}, k::Int, t) where {TG, T} = ϕ⁽ᵏ⁾(G.G, k + G.p, G.sJ + t) / G.den
ϕ⁽ᵏ⁾⁻¹(G::TiltedGenerator{TG, T}, k::Int, y; start_at = G.sJ) where {TG, T} = ϕ⁽ᵏ⁾⁻¹(G.G, k + G.p, y * G.den; start_at = start_at+G.sJ) - G.sJ
ϕ⁽¹⁾(G::TiltedGenerator{TG, T}, t) where {TG, T} = ϕ⁽ᵏ⁾(G, 1, t)
Distributions.params(G::TiltedGenerator) = (Distributions.params(G.G)..., sJ = G.sJ)



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
frailty(::Generator) = nothing
max_monotony(::AbstractFrailtyGenerator) = Inf
ϕ(G::AbstractFrailtyGenerator, t) = Distributions.mgf(frailty(G), -t)
𝒲₋₁(G::AbstractFrailtyGenerator, d::Int) = WilliamsonFromFrailty(frailty(G), d)

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
