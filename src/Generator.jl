"""
    Generator

Abstract representation of an Archimedean generator. A generator is a decreasing
function `ϕ : [0,∞) → [0,1]` with `ϕ(0)=1` and `ϕ(∞)=0`; constructing a
`d`-dimensional Archimedean copula additionally requires the appropriate
`d`-monotonicity.

`Generator` is a supported public extension point. A downstream generator `G`
must implement:

- `ϕ(G, t)`, the mathematical generator;
- `max_monotony(G)`, the largest supported Williamson order (`Inf` for a
  completely monotone generator);
- `Distributions.params(G)`, returning a `NamedTuple` of its public parameters.

These methods are sufficient to construct `ArchimedeanCopula(d, G)` and use its
generic CDF path: the inverse of `ϕ` is obtained numerically when no specialized
method exists. Other operations can require more. Generic automatic
differentiation and inverse-Williamson fallbacks provide density and sampling
for suitably regular generators, but their numerical success is not implied by
the three-method contract alone, especially at singularities and parameter
boundaries.

Only this mathematical interface is public. Copulas.jl's generator subtype
hierarchy beyond documented public types, derivative and inverse hooks, radial
caches, fitting hooks, dispatch traits, and specialized numerical machinery are
implementation details. The developer guide describes those optional in-package
optimizations separately.

See also: [`ArchimedeanCopula`](@ref), [`ϕ`](@ref),
[`max_monotony`](@ref), [`WilliamsonGenerator`](@ref),
[`FrailtyGenerator`](@ref),
[`Distributions.params`](@extref Distributions Distributions.params).
"""
abstract type Generator end
Base.eltype(G::Generator) = _sample_eltype(G)
function (TG::Type{<:Generator})(args...;kwargs...)
    S = hasproperty(TG, :body) ? TG.body : TG
    T = S.name.wrapper 
    return T(args..., values(kwargs)...)
end
Base.broadcastable(x::Generator) = Ref(x)
_parameter_dof(x::Generator) = _parameter_dof(Distributions.params(x))

"""
    max_monotony(G::Generator)

Return the largest Williamson order for which `G` is known to be monotone.
`Inf` denotes complete monotonicity. This public mathematical query is used to
validate the dimensions of Archimedean and Liouville constructions.

See also: [`Generator`](@ref), [`ArchimedeanCopula`](@ref),
[`LiouvilleCopula`](@ref), [`ϕ`](@ref).
"""
max_monotony(G::Generator) = throw("This generator does not have a defined max monotony. You need to implement `max_monotony(G)`.")

"""
    ϕ(G::Generator, t)
    ϕ(G::Generator)

Evaluate the Archimedean generator at `t ≥ 0`, or return its callable unary
form. A valid implementation is decreasing, satisfies `ϕ(G, 0) = 1`, tends to
zero at infinity, and has the monotonicity reported by `max_monotony(G)`.

See also: [`Generator`](@ref), [`max_monotony`](@ref),
[`ArchimedeanCopula`](@ref), [`WilliamsonGenerator`](@ref).
"""
ϕ(   G::Generator, t) = throw("This generator has not been defined correctly, the function `ϕ(G,t)` is not defined.")
ϕ(G::Generator) = Base.Fix1(ϕ,G)

"""
    ϕ⁻¹(G::Generator, u)

Return the generalized inverse of `ϕ(G, ·)` at `u ∈ [0,1]`. The generic
internal fallback uses scalar root finding; generator implementations may
specialize it for accuracy, boundary behavior, or performance.

See also: [`ϕ`](@ref), [`ϕ⁽¹⁾`](@ref), [`Generator`](@ref).
"""
ϕ⁻¹( G::Generator, x) = Roots.find_zero(t -> ϕ(G,t) - x, (0.0, Inf))

"""
    ϕ⁽¹⁾(G::Generator, t)

Evaluate the first derivative of the generator. The generic internal fallback
uses forward-mode automatic differentiation. Specialized methods must preserve
the derivative of `ϕ`, including its sign and limiting behavior.

See also: [`ϕ`](@ref), [`ϕ⁻¹⁽¹⁾`](@ref), [`ϕ⁽ᵏ⁾`](@ref).
"""
ϕ⁽¹⁾(G::Generator, t) = ForwardDiff.derivative(x -> ϕ(G,x), t)

"""
    ϕ⁻¹⁽¹⁾(G::Generator, u)

Evaluate the derivative of the inverse generator through
`1 / ϕ⁽¹⁾(G, ϕ⁻¹(G, u))`. This is an internal conditioning and sampling hook;
specializations must agree with that identity wherever the inverse is regular.

See also: [`ϕ⁻¹`](@ref), [`ϕ⁽¹⁾`](@ref), [`distortion`](@ref).
"""
ϕ⁻¹⁽¹⁾(G::Generator, t) = inv(ϕ⁽¹⁾(G, ϕ⁻¹(G, t)))

"""
    ϕ⁽ᵏ⁾(G::Generator, k::Int, t)

Evaluate the derivative of order `k ≥ 0`. The generic internal fallback uses a
Taylor expansion. A specialization is a numerical fast path and must return
the same derivative, with `k = 0` corresponding to `ϕ(G, t)`.

See also: [`ϕ`](@ref), [`ϕ⁽¹⁾`](@ref), [`ϕ⁽ᵏ⁾⁻¹`](@ref).
"""
function ϕ⁽ᵏ⁾(G::Generator, k::Int, t)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    return _mul_factorial(taylor(ϕ(G), t, k)[end], k)
end

"""
    ϕ⁽ᵏ⁾⁻¹(G::Generator, k::Int, y; start_at=y)

Invert the `k`th generator derivative on the relevant monotone branch. The
generic internal fallback expands a positive bracket and applies bisection.
`start_at` identifies the lower branch boundary used by tilted generators.

See also: [`ϕ⁽ᵏ⁾`](@ref), [`ϕ⁻¹`](@ref), [`Generator`](@ref).
"""
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

abstract type MarkerGenerator <: Generator end

"""
    IndependentGenerator()

Parameter-free Archimedean generator `ϕ(t) = exp(-t)`, corresponding to the
independence copula in every dimension. It is useful when composing generic
generator-based models; ordinary users will usually construct
`IndependentCopula` directly.

See also: [`IndependentCopula`](@ref), [`Generator`](@ref),
[`ArchimedeanCopula`](@ref).
"""
struct IndependentGenerator <: MarkerGenerator end
struct MGenerator <: MarkerGenerator end
struct WGenerator <: MarkerGenerator end

Distributions.params(::MarkerGenerator) = (;)

"""
    limit_kind(component, ::Val{d})

Classify whether a generator, tail, or composite component is exactly at a
canonical dependence limit in dimension `d`. Internal constructors and
algorithms use the result to preserve independence, comonotonicity, or the
bivariate lower bound without relying on approximate parameter comparisons.
Families return `NO_LIMIT` away from those values. This protocol is not public
API.

See also: [`LimitKind`](@ref), [`CopulaMeasureStyle`](@ref),
[`Generator`](@ref), [`Tail`](@ref).
"""
@inline limit_kind(::Generator, ::Val) = NO_LIMIT
@inline limit_kind(::MGenerator, ::Val) = M_LIMIT
@inline limit_kind(::WGenerator, ::Val) = W_LIMIT
@inline limit_kind(::IndependentGenerator, ::Val) = Π_LIMIT

max_monotony(::IndependentGenerator) = Inf
max_monotony(::MGenerator) = Inf
max_monotony(::WGenerator) = 2

ϕ(::IndependentGenerator, t) = exp(-t)
ϕ⁻¹(::IndependentGenerator, u) = -log(u)
ϕ⁽¹⁾(::IndependentGenerator, t) = -exp(-t)
ϕ⁻¹⁽¹⁾(::IndependentGenerator, u) = -inv(u)
ϕ⁽ᵏ⁾(::IndependentGenerator, k::Int, t) = (-1)^k * exp(-t)
ϕ⁽ᵏ⁾⁻¹(::IndependentGenerator, ::Int, u; start_at=u) =
    iszero(u) ? oftype(float(u), Inf) : -log(abs(u))
frailty(::IndependentGenerator) = Distributions.Dirac(1.0)

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

For integer ``d ≥ 2``, a ``d``-monotone Archimedean generator ``\\phi`` has these properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``(-1)^{d-2}\\phi^{(d-2)}`` is non-increasing and convex.

For such a function ``\\phi``, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``X``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\\phi)(x) = 1 - \\frac{(-x)^{d-1} \\phi_+^{(d-1)}(x)}{(d-1)!} - \\sum_{k=0}^{d-2} \\frac{(-x)^k \\phi^{(k)}(x)}{k!}
```

The result is the corresponding non-negative univariate distribution, not a
particular concrete distribution type. It need not be continuous: an inverse
at the original Williamson order can return the original discrete radial law.
Use `Distributions.cdf` and `rand` to evaluate its CDF and sample it. Density or
mass evaluation follows the returned distribution's supported interface; no
Lebesgue density is promised for discrete or singular laws.

References: 
    - Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    - McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.

See also: [`WilliamsonGenerator`](@ref), [`Generator`](@ref),
[`ArchimedeanCopula`](@ref), [`LiouvilleCopula`](@ref).
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
    n <= max_monotony(G) || throw(ArgumentError("cannot invert a generator of maximal monotonicity $(max_monotony(G)) at order $d"))
    isinteger(d) && return 𝒲₋₁(G, n)
    return WilliamsonBetaProduct(𝒲₋₁(G, n), Distributions.Beta(d, n - d), n)
end

𝒲₋₁(::IndependentGenerator, d::Integer) = Distributions.Gamma(d, 1)
𝒲₋₁(::IndependentGenerator, d::Real) = Distributions.Gamma(d, 1)

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
Distributions.rand(rng::Distributions.AbstractRNG, dist::𝒲₋₁) =
    Distributions.quantile(dist, rand(rng))
Base.minimum(::𝒲₋₁) = 0.0
Base.maximum(::𝒲₋₁) = Inf
function Distributions.quantile(dist::𝒲₋₁, p::Real)
    return _quantile_from_cdf(dist, p)
end

include("UnivariateDistribution/Radials/WilliamsonBetaProduct.jl")

"""
    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)
    WilliamsonGenerator(atoms::AbstractVector, weights::AbstractVector, d)
    𝒲(X::Distributions.UnivariateDistribution,d)
    𝒲(atoms::AbstractVector, weights::AbstractVector, d)

The `𝒲` type (also available as `WilliamsonGenerator`) constructs a d-monotonous archimedean generator from a positive random variable `X::Distributions.UnivariateDistribution`. The transformation is implemented fully generically in the package.

For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and a positive real order ``d``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

For integer ``d ≥ 2`` and a strictly positive radial variable, this function has
the following properties:
- We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``(-1)^{d-2}\\phi^{(d-2)}`` is non-increasing and convex.

These properties characterize a *d-monotone Archimedean generator*. Real orders
are also supported, but the integer derivative characterization above should
not be read as a definition of fractional derivatives. Copula dimensions remain
integers and are checked against the supported order. The function is accessed by

    G = WilliamsonGenerator(X, d)
    ϕ(G,t)

Note that you'll always have:

    max_monotony(WilliamsonGenerator(X,d)) == d


Special case (finite-support discrete X)

- For a finite discrete radial law, the transform is
  `ϕ(t) = ∑_j w_j · (1 − t/r_j)_+^(d−1)`. It is piecewise polynomial for
  integer orders; real orders need not give polynomials.
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
Distributions.params(G::𝒲) = (X=G.X, order=G.order)
max_monotony(G::𝒲) = G.order

_williamson_primal(t) = t
_williamson_primal(t::ForwardDiff.Dual) = ForwardDiff.value(t)

"""
Generic fallback for ϕ on WilliamsonGenerator (non-discrete-nonparametric TX).
Specializations for `TX<:DiscreteNonParametric` are provided below.
"""
function _williamson_tail_expectation(f, X::Distributions.ContinuousUnivariateDistribution, t)
    a = _williamson_primal(t)
    p = Distributions.ccdf(X, a)
    iszero(p) && return zero(float(t))
    Xt = Distributions.truncated(X, a, Inf)
    return p * Distributions.expectation(f, Xt)
end
function ϕ(G::𝒲, t)
    t <= 0 && return one(t)
    if G.X isa Distributions.ContinuousUnivariateDistribution
        return _williamson_tail_expectation(G.X, t) do y
            (1 - t / y)^(G.order - 1)
        end
    end
    return Distributions.expectation(y -> (y > t) ? (1 - t / y)^(G.order - 1) : zero(t), G.X)
end

function ϕ⁽ᵏ⁾(G::𝒲, k::Int, t)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    k == 0 && return ϕ(G, t)
    t < 0 && return zero(float(t))

    if k < G.order
        coefficient = _falling_factorial(G.order - 1, k)
        value = if G.X isa Distributions.ContinuousUnivariateDistribution
            _williamson_tail_expectation(G.X, t) do y
                (1 - t / y)^(G.order - 1 - k) / y^k
            end
        else
            Distributions.expectation(G.X) do y
                y > t ? (1 - t / y)^(G.order - 1 - k) / y^k : zero(t + y + G.order)
            end
        end
        return (isodd(k) ? -coefficient : coefficient) * value
    end

    if k == G.order &&
       G.X isa Distributions.ContinuousUnivariateDistribution &&
       t > 0
        value = factorial(k - 1) * Distributions.pdf(G.X, t) / t^(k - 1)
        return isodd(k) ? -value : value
    end

    return invoke(ϕ⁽ᵏ⁾, Tuple{Generator, Int, Any}, G, k, t)
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

distortion_measure_style(D::ArchimedeanDistortion{<:WilliamsonGenerator}) = archimedean_measure_style(D.G, Val(D.p + 1))
function Distributions.quantile(
    D::ArchimedeanDistortion{<:WilliamsonGenerator},
    α::Real,
)
    distortion_measure_style(D) isa NonAbsolutelyContinuousMeasure &&
        return _quantile_from_cdf(D, α)
    return invoke(
        Distributions.quantile,
        Tuple{ArchimedeanDistortion,Real},
        D,
        α,
    )
end

# Exact inverse paths when the forward transform retains its radial law.
function _williamson_inverse_preserved(G::𝒲, d::Real)
    isfinite(d) && d > 0 || throw(ArgumentError("the Williamson order must be finite and positive"))
    d == G.order && return G.X
    d < G.order && return WilliamsonBetaProduct(G.X, Distributions.Beta(d, G.order - d), G.order)
    throw(ArgumentError("cannot invert a Williamson transform above its source order $(G.order)"))
end
𝒲₋₁(G::𝒲, d::Integer) = _williamson_inverse_preserved(G, d)
𝒲₋₁(G::𝒲, d::Real) = _williamson_inverse_preserved(G, d)
function 𝒲(X::𝒲₋₁, d::Real)
    d == X.order && return X.G
    return invoke(𝒲, Tuple{Any, Real}, X, d)
end
function 𝒲(X::WilliamsonBetaProduct, d::Real)
    target_order = first(Distributions.params(X.B))
    d == target_order && return 𝒲(X.X, X.source_order)
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
    EmpiricalGenerator(u::AbstractMatrix; pseudo_values=true)

Nonparametric Archimedean generator fit via inversion of the empirical Kendall distribution.

It returns a [`Generator`](@ref) representing the fitted generator. Its concrete
representation is an implementation detail and may depend on the data.

Usage

    G = EmpiricalGenerator(u)

where `u::AbstractMatrix` has one component per row and one observation per
column (`d×n`). With `pseudo_values=true`, values must already be
pseudo-observations; pass `pseudo_values=false` to rank-transform raw data.

Notes
* The recovered discrete radial support is rescaled so its largest atom equals 1 (scale is not identifiable).
* Code should use the documented `Generator` operations rather than rely on a
  particular concrete return type.

References
* [mcneil2009](@cite)
* [williamson1956](@cite)
* [genest2011a](@cite) Genest, Neslehova and Ziegel (2011), Inference in Multivariate Archimedean Copula Models
"""
function EmpiricalGenerator(u::AbstractMatrix; pseudo_values=true)
    d = size(u, 1)
    U = pseudo_values ? u : pseudos(u)
    W = _kendall_sample(U)
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
    FrailtyGenerator(D)

Construct a completely monotone Archimedean generator from a non-negative
continuous frailty distribution `D`. Its generator is the Laplace transform

```math
\\phi(t)=\\mathbb{E}[e^{-tV}]=\\operatorname{mgf}_D(-t), \\qquad V\\sim D.
```

`D` must have non-negative support and implement `Distributions.mgf`.
Generator derivatives additionally use expectations of `V^k exp(-tV)`, and
sampling an associated Archimedean copula uses `rand` on the frailty. The
resulting complete monotonicity permits construction in every dimension.

Multiplying `V` by a positive constant changes the generator scale but not the
resulting Archimedean copula. Consequently the frailty distribution is not an
identifiable copula parameterization without a scale convention. This generic
wrapper is useful for custom frailties; named generator families generally
offer clearer parameter validation and fitting support.

# Example
```julia
G = FrailtyGenerator(Gamma(2.0, 1.0))
C = ArchimedeanCopula(3, G)
```

References:
* [hofert2009](@cite) M. Hofert (2009). Efficiently sampling Archimedean copulas.

See also: [`Generator`](@ref), [`ArchimedeanCopula`](@ref), [`ϕ`](@ref),
[`WilliamsonGenerator`](@ref).
"""
FrailtyGenerator

abstract type AbstractFrailtyGenerator<:Generator end
frailty(::Generator) = nothing
max_monotony(::AbstractFrailtyGenerator) = Inf
ϕ(G::AbstractFrailtyGenerator, t) = Distributions.mgf(frailty(G), -t)
function ϕ⁽ᵏ⁾(G::AbstractFrailtyGenerator, k::Int, t)
    k >= 0 || throw(ArgumentError("k must be non-negative"))
    k == 0 && return ϕ(G, t)
    value = Distributions.expectation(frailty(G)) do v
        v^k * exp(-t * v)
    end
    return isodd(k) ? -value : value
end
𝒲₋₁(G::AbstractFrailtyGenerator, d::Int) = WilliamsonFromFrailty(frailty(G), d)

struct FrailtyGenerator{TF}<:AbstractFrailtyGenerator
    F::TF
    function FrailtyGenerator(F::Distributions.ContinuousUnivariateDistribution)
        @assert Base.minimum(F) >= 0
        return new{typeof(F)}(F)
    end
end
Distributions.params(G::FrailtyGenerator) = (F=G.F,)
frailty(G::FrailtyGenerator) = G.F

"""
    AbstractUnivariateGenerator <: Generator

Internal capability type for parametric generators whose user-facing
parameters are represented by a single univariate generator object. It is used
to share constructor and fitting machinery; downstream packages must not rely
on this subtype as a stable extension interface.

See also: [`Generator`](@ref), [`FrailtyGenerator`](@ref),
[`_available_fitting_methods`](@ref).
"""
abstract type AbstractUnivariateGenerator <: Generator end
abstract type AbstractUnivariateFrailtyGenerator <: AbstractFrailtyGenerator end
const UnivariateGenerator = Union{AbstractUnivariateGenerator,AbstractUnivariateFrailtyGenerator}
