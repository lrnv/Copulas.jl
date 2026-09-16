
###############################################################################
#####  Conditioning framework.
#####  User-facing function: `condition(), rosenblatt(), inverse_rosenblatt()`
#####
#####  Internal extension points for model-specific conditioning:
#####   - `distortion(C, js, uⱼₛ, i)`
#####   - `conditional_copula(C, js, uⱼₛ)`
#####  Their generic methods construct `DistortionFromCop` and
#####  `ConditionalCopula`, respectively.
###############################################################################

# A few utilities :
function _assemble(D, is, js, uᵢₛ, uⱼₛ)
    Tᵢ = eltype(typeof(uᵢₛ)); Tⱼ = eltype(typeof(uⱼₛ)); T = promote_type(Tᵢ, Tⱼ)
    w = fill(one(T), D)
    @inbounds for (k,i) in pairs(is); w[i] = uᵢₛ[k]; end
    @inbounds for (k,j) in pairs(js); w[j] = uⱼₛ[k]; end
    return w
end

# Generic fallbacks. Family implementations specialize these lowercase hooks;
# the concrete types remain implementation details of the generic path.
"""
    distortion(C::Copula, js, ujs, i)

Return the uniform-scale conditional marginal of coordinate `i` given
`U[js] = ujs`. This internal extension hook defaults to `DistortionFromCop`,
which uses mixed CDF partials. Family specializations may provide a faster or
atom-aware distribution but must preserve the same coordinate ordering, scale,
and conditional-law semantics. Use public `condition` in downstream code.

See also: [`conditional_copula`](@ref), [`_partial_cdf`](@ref),
[`Distortion`](@ref), [`condition`](@ref).
"""
distortion(C::Copula, js, uⱼₛ, i) = DistortionFromCop(C, js, uⱼₛ, i)

"""
    _partial_cdf(C, is, js, uis, ujs)

Evaluate the mixed derivative of the copula CDF with respect to coordinates
`js`, at the point assembled from free coordinates `is => uis`, conditioned
coordinates `js => ujs`, and ones elsewhere. This internal signed sub-density
is the common denominator and numerator primitive for generic conditioning.
The fallback uses automatic differentiation; a specialization is required when
the numerical CDF cannot accept dual numbers or when singular semantics demand
an exact implementation.

See also: [`_mixed_partial`](@ref), [`distortion`](@ref),
[`conditional_copula`](@ref).
"""
_partial_cdf(C, is, js, uᵢₛ, uⱼₛ) = _mixed_partial(u -> Distributions.cdf(C, u),_assemble(length(C), is, js, uᵢₛ, uⱼₛ), js,)

"""
    _box_partial_cdf(C, is, ps, bs, uis, ups, lo, hi)

Evaluate the mixed derivative with respect to the point coordinates `ps` of the
C-volume of the box `∏ₖ [lo[k], hi[k]]` on coordinates `bs`, at free coordinates
`is => uis`, point coordinates `ps => ups`, and ones elsewhere. It is the
inclusion–exclusion sum over the `2^q` corners of the box of [`_partial_cdf`](@ref)
with the box coordinates placed among the free ones, so `q == 0` is `_partial_cdf`
and `ps == ()` is the C-volume computed by [`measure`](@ref). This internal
primitive is the numerator and denominator of interval conditioning, and the
likelihood of a `SklarDist` with atoms.

See also: [`_partial_cdf`](@ref), [`measure`](@ref), [`condition`](@ref).
"""
function _box_partial_cdf(C, is, ps, bs::NTuple{q,Int}, uᵢₛ, uₚₛ, lo, hi) where {q}
    q == 0 && return _partial_cdf(C, is, ps, uᵢₛ, uₚₛ)
    corner(mask) = ntuple(k -> (mask >> (k - 1)) & 1 == 1 ? hi[k] : lo[k], q)
    term(mask) = _partial_cdf(C, (is..., bs...), ps, promote(uᵢₛ..., corner(mask)...), uₚₛ)
    # The corner at `hi` carries a plus sign and each bound moved to `lo` flips it,
    # so the all-`lo` corner (mask 0) carries (-1)^q.
    r = isodd(q) ? -term(0) : term(0)
    for mask in 1:(1 << q) - 1
        t = term(mask)
        r = isodd(q - count_ones(mask)) ? r - t : r + t
    end
    return r
end
# With no point coordinate the sum is a C-volume, and `measure` computes it
# over the same corners with fewer allocations (the box on `bs`, the free
# coordinates as `[0, u]`, and `[0, 1]` elsewhere).
function _box_partial_cdf(C::Copula{D}, is, ps::Tuple{}, bs::NTuple{q,Int}, uᵢₛ, uₚₛ, lo, hi) where {D,q}
    q == 0 && return _partial_cdf(C, is, ps, uᵢₛ, uₚₛ)
    T = promote_type(eltype(uᵢₛ), eltype(lo), eltype(hi))
    lower = ntuple(D) do j
        k = findfirst(==(j), bs)
        k === nothing ? zero(T) : T(lo[k])
    end
    upper = ntuple(D) do j
        k = findfirst(==(j), bs)
        k === nothing || return T(hi[k])
        k = findfirst(==(j), is)
        k === nothing ? one(T) : T(uᵢₛ[k])
    end
    return measure(C, lower, upper)
end

_process_tuples(::Val{D}, js::NTuple{p, Int64}, ujs::NTuple{p, Float64}) where {D,p} = (js, ujs)
_process_tuples(::Val{D}, j::Int64, uj::Real) where {D} = ((j,), (uj,))
function _process_tuples(::Val{D}, js, ujs) where D
    p, p2 = length(js), length(ujs)
    @assert 0 < p < D "js=$(js) must be a non-empty proper subset of 1:D of length at most D-1 (D = $D)"
    @assert p == p2 "uⱼₛ length must match js length"
    jst = Tuple(collect(Int, js))
    @assert all(in(1:D), jst)
    ujst = Tuple(collect(float.(ujs)))
    return (jst, ujst)
end

# Normalise the arguments of the interval form of `condition`: index tuple,
# homogeneous bound tuples, and the validation 0 ≤ lo ≤ hi ≤ 1.
function _process_intervals(::Val{D}, js, lo, hi) where {D}
    jst = js isa Integer ? (Int(js),) : Tuple(collect(Int, js))
    lot = lo isa Real ? (lo,) : promote(lo...)
    hit = hi isa Real ? (hi,) : promote(hi...)
    p = length(jst)
    0 < p < D || throw(ArgumentError("js=$(js) must be a non-empty proper subset of 1:$D"))
    all(in(1:D), jst) && allunique(jst) ||
        throw(ArgumentError("js=$(js) must be distinct indices in 1:$D"))
    length(lot) == p && length(hit) == p ||
        throw(ArgumentError("lo and hi must have one bound per conditioned coordinate"))
    all(k -> zero(lot[k]) <= lot[k] <= hit[k] <= one(hit[k]), 1:p) ||
        throw(ArgumentError("interval bounds must satisfy 0 ≤ lo ≤ hi ≤ 1"))
    return jst, lot, hit
end

# The latent uniform interval (F(x⁻), F(x)] of an observation `x` under margin
# `m`. A continuous margin observes a point, so both ends coincide; an atom
# spans the jump of the CDF, computed as a CDF/PMF difference because
# `cdf(m, prevfloat(x))` is not a left limit for every discrete family.
_latent_interval(m::Distributions.UnivariateDistribution, x) =
    _latent_interval(Distributions.value_support(typeof(m)), m, x)
function _latent_interval(::Type{Distributions.Continuous}, m, x)
    u = Distributions.cdf(m, x)
    return (u, u)
end
function _latent_interval(::Type{Distributions.Discrete}, m, x)
    hi = Distributions.cdf(m, x)
    lo = hi - Distributions.pdf(m, x)
    return (max(lo, zero(lo)), hi)
end
_has_atoms(margins::Tuple) =
    any(m -> Distributions.value_support(typeof(m)) === Distributions.Discrete, margins)

"""
    Distortion <: Distributions.ContinuousUnivariateDistribution

Abstract super-type for objects describing the (uniform-scale) conditional marginal
transformation U_i | U_J = u_J of a copula.

Subtypes implement cdf/quantile on [0,1]. They are not full arbitrary distributions;
they model how a uniform variable is distorted by conditioning. They can be applied
as a function to a base marginal distribution to obtain the conditional marginal on
the original scale: if `D::Distortion` and `X::UnivariateDistribution`, then `D(X)`
is the distribution of `X_i | U_J = u_J`.

See also: [`distortion`](@ref), [`DistortedDist`](@ref), [`condition`](@ref).
"""
abstract type Distortion<:Distributions.ContinuousUnivariateDistribution end

quantile_strategy(::Type{<:Distortion}) = LogCDFQuantile()

# Some exact conditional laws are atomic despite the historical continuous
# supertype of `Distortion`. Keep that semantic capability explicit so callers
# need not infer it from concrete implementation names.
distortion_measure_style(::Type{<:Distortion}) = AbsolutelyContinuousMeasure()
distortion_measure_style(D::Distortion) = distortion_measure_style(typeof(D))

(D::Distortion)(::Distributions.Uniform) = D
(D::Distortion)(X::Distributions.UnivariateDistribution) = DistortedDist(D, X)
Distributions.minimum(::Distortion) = 0.0
Distributions.maximum(::Distortion) = 1.0
Distributions.quantile(d::Distortion, α::Real) = _quantile_from_cdf(d, α)
# You have to implement a cdf, and you can implement a pdf, either in log scaleor not:
Distributions.logcdf(d::Distortion, t::Real) = log(Distributions.cdf(d, t))
Distributions.cdf(d::Distortion, t::Real) = exp(Distributions.logcdf(d, t))
function Distributions.logpdf(d::Distortion, u::Real)
    (0.0 <= u <= 1.0) || return -Inf
    v = ForwardDiff.derivative(t -> Distributions.cdf(d, t), float(u))
    v <= 0 && return -Inf
    return log(v)
end
Distributions.pdf(d::Distortion, t::Real) = exp(Distributions.logpdf(d, t))

"""
    DistortionFromCop{TC,p,q,T} <: Distortion

Generic, uniform-scale conditional marginal transformation for a copula.

This is the default fallback (based on mixed partial derivatives computed via
automatic differentiation) used when a faster specialized `Distortion` is not
available for a given copula family. The conditioning event is the intersection
of a point condition `U_js = ujs` and a box condition `U_bs ∈ ∏ₖ [lo[k], hi[k]]`;
either part may be empty.

Parameters
- `TC`: copula type
- `p`: length of the point-conditioned index set J (static)
- `q`: length of the interval-conditioned index set B (static)
- `T`: element type for the conditioned values u_J and the bounds

Construction
- `DistortionFromCop(C::Copula, js::NTuple{p,Int}, ujs::NTuple{p,<:Real}, i::Int)`
    builds the distortion for the conditional marginal of index `i` given `U_js = ujs`.
- `DistortionFromCop(C::Copula, js, ujs, bs::NTuple{q,Int}, lo::NTuple{q,<:Real}, hi::NTuple{q,<:Real}, i::Int)`
    additionally conditions on `U_bs ∈ ∏ₖ [lo[k], hi[k]]`; the denominator is then a
    probability rather than a density, and it throws an `ArgumentError` when the
    event has zero probability.

Notes
- A convenience method `DistortionFromCop(C, j::Int, uj::Real, i::Int)` exists for
    the common `p = 1` case.
"""
struct DistortionFromCop{TC,p,q,T}<:Distortion
    C::TC
    i::Int
    js::NTuple{p,Int}
    uⱼₛ::NTuple{p,T}
    bs::NTuple{q,Int}
    lo::NTuple{q,T}
    hi::NTuple{q,T}
    den::T
    function DistortionFromCop(C::Copula{D}, js, uⱼₛ, i) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        p = length(jst)
        den = p==1 ? Distributions.pdf(subsetdims(C, jst), uⱼₛt[1]) :
                     Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        T = promote_type(eltype(uⱼₛt), typeof(den))
        return new{typeof(C), p, 0, T}(C, i, jst, NTuple{p,T}(uⱼₛt), (), (), (), T(den))
    end
    function DistortionFromCop(C::Copula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real},
                               bs::NTuple{q,Int}, lo::NTuple{q,<:Real}, hi::NTuple{q,<:Real},
                               i::Int) where {D,p,q}
        q == 0 && return DistortionFromCop(C, js, uⱼₛ, i)
        den = _box_partial_cdf(C, (), js, bs, (), uⱼₛ, lo, hi)
        den > zero(den) ||
            throw(ArgumentError("the conditioning event has zero probability under the copula"))
        T = promote_type(eltype(uⱼₛ), eltype(lo), eltype(hi), typeof(den))
        return new{typeof(C), p, q, T}(C, i, js, NTuple{p,T}(uⱼₛ), bs,
                                       NTuple{q,T}(lo), NTuple{q,T}(hi), T(den))
    end
end
function Distributions.cdf(d::DistortionFromCop, u::Real)
    T = promote_type(typeof(u), typeof(d.den))
    u <= 0 && return zero(T)
    u >= 1 && return one(T)
    return _box_partial_cdf(d.C, (d.i,), d.js, d.bs, (T(u),), d.uⱼₛ, d.lo, d.hi) / d.den
end

# Density on the uniform scale: f_{i|J}(u | u_J) = (∂^{p+1} C / ∂(J..., i))(u, u_J) / (∂^{p} C / ∂J)(1, u_J)
function Distributions.logpdf(d::DistortionFromCop, u::Real)
    # Support checks
    (0 < u < 1) || return -Inf
    d.den <= 0 && return -Inf

    # Mixed partial derivative of order p+1 w.r.t. (J..., i). Going through
    # `_partial_cdf` lets models provide this quantity without differentiating
    # their numerical CDF implementation.
    num = _box_partial_cdf(
        d.C,
        (),
        (d.js..., d.i),
        d.bs,
        (),
        promote(d.uⱼₛ..., u),
        d.lo,
        d.hi,
    )
    (num <= 0 || !isfinite(num)) && return -Inf
    return log(num) - log(d.den)
end

"""
    DistortedDist{Disto,Distrib} <: Distributions.UnivariateDistribution

Internal representation of a conditioned marginal on its original scale.
`D` describes the conditional law on the uniform scale and `X` is the original
univariate marginal. Consequently its CDF is `D(cdf(X, x))`, while quantiles
apply the two generalized inverses in reverse order. This representation is
used when conditioning a `SklarDist`; its storage fields are not public API.

See also: [`Distortion`](@ref), [`condition`](@ref), [`SklarDist`](@ref).
"""
struct DistortedDist{Disto, Distrib}<:Distributions.ContinuousUnivariateDistribution
    D::Disto
    X::Distrib
    function DistortedDist(D::Distortion, X::Distributions.UnivariateDistribution)
        return new{typeof(D), typeof(X)}(D, X)
    end
end
Base.minimum(D::DistortedDist) = minimum(D.X)
Base.maximum(D::DistortedDist) = maximum(D.X)
Distributions.cdf(D::DistortedDist, t::Real) = Distributions.cdf(D.D, Distributions.cdf(D.X, t))
Distributions.logcdf(D::DistortedDist, t::Real) = Distributions.logcdf(D.D, Distributions.cdf(D.X, t))
Distributions.quantile(D::DistortedDist, α::Real) = Distributions.quantile(D.X, Distributions.quantile(D.D, α))
function Distributions.logpdf(D::DistortedDist, t::Real)
    lo, hi = _latent_interval(D.X, t)
    # A continuous margin: density of `X` times the distortion density at `F(t)`.
    lo == hi && return Distributions.logpdf(D.X, t) + Distributions.logpdf(D.D, hi)
    # An atom of `X`: the conditional probability mass is the distortion's
    # probability of the latent interval (F(t⁻), F(t)].
    mass = Distributions.cdf(D.D, hi) - Distributions.cdf(D.D, lo)
    return mass > zero(mass) ? log(mass) : oftype(mass, -Inf)
end

"""
    ConditionalCopula{d} <: Copula{d}

Internal fallback for the copula of the remaining coordinates
`U_I | U_J = u_J`. It computes each conditional marginal distortion and uses
mixed CDF partials to normalize the joint conditional law. Coordinates in `I`
retain their natural order. Family-specific `conditional_copula` methods may
replace this representation, so its fields are not a downstream contract.

See also: [`conditional_copula`](@ref), [`distortion`](@ref),
[`_partial_cdf`](@ref), [`condition`](@ref).
"""
struct ConditionalCopula{d, D, p, q, T, TDs}<:Copula{d}
    C::Copula{D}
    js::NTuple{p, Int}
    is::NTuple{d, Int}
    uⱼₛ::NTuple{p, T}
    bs::NTuple{q, Int}
    lo::NTuple{q, T}
    hi::NTuple{q, T}
    den::T
    logden::T
    distortions::TDs
    function ConditionalCopula(C::Copula{D}, js, uⱼₛ) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        ist = Tuple(i for i in 1:D if i ∉ jst)
        p = length(jst)
        d = D - p
        distos = Tuple(distortion(C, jst, uⱼₛt, i) for i in ist)
        den = all(disto -> disto isa DistortionFromCop, distos) ? distos[1].den :
              (p==1 ? Distributions.pdf(subsetdims(C, jst), uⱼₛt[1]) :
                      Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt)))
        T = promote_type(eltype(uⱼₛt), typeof(den))
        denT = T(den)
        return new{d, D, p, 0, T, typeof(distos)}(
            C, jst, ist, NTuple{p,T}(uⱼₛt), (), (), (), denT,
            denT > zero(T) ? log(denT) : T(-Inf), distos
        )
    end
    # Conditional copula of `U_I | U_js = uⱼₛ, U_bs ∈ ∏ₖ [lo[k], hi[k]]`.
    function ConditionalCopula(C::Copula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real},
                               bs::NTuple{q,Int}, lo::NTuple{q,<:Real},
                               hi::NTuple{q,<:Real}) where {D,p,q}
        q == 0 && return ConditionalCopula(C, js, uⱼₛ)
        ist = Tuple(i for i in 1:D if i ∉ js && i ∉ bs)
        d = D - p - q
        distos = Tuple(DistortionFromCop(C, js, uⱼₛ, bs, lo, hi, i) for i in ist)
        den = distos[1].den
        T = typeof(den)
        return new{d, D, p, q, T, typeof(distos)}(
            C, js, ist, NTuple{p,T}(uⱼₛ), bs, NTuple{q,T}(lo), NTuple{q,T}(hi),
            den, log(den), distos
        )
    end
end
Base.eltype(::ConditionalCopula{d,D,p,q,T}) where {d,D,p,q,T} = T

"""
    conditional_copula(C::Copula, js, ujs)

Return the copula of the remaining coordinates conditional on `U[js] = ujs`.
The internal fallback builds a `ConditionalCopula` from mixed CDF partials and
the marginal `distortion`s. A family specialization may provide a simpler or
faster representation, but must preserve the remaining coordinates' natural
order and the same conditional law. Public code should call `condition`.

See also: [`ConditionalCopula`](@ref), [`distortion`](@ref),
[`_partial_cdf`](@ref), [`condition`](@ref).
"""
conditional_copula(C::Copula, js, uⱼₛ) = ConditionalCopula(C, js, uⱼₛ)

# The distortion of coordinate `i` given points on `js` and a box on `bs`: the
# family hook when the box is empty, the generic box form otherwise.
_distortion_box(C::Copula, js, uⱼₛ, bs::Tuple{}, lo::Tuple{}, hi::Tuple{}, i::Int) =
    distortion(C, js, uⱼₛ, i)
_distortion_box(C::Copula, js, uⱼₛ, bs, lo, hi, i::Int) =
    DistortionFromCop(C, js, uⱼₛ, bs, lo, hi, i)
function _cdf(CC::ConditionalCopula{d,D,p,q,T}, v::AbstractVector{<:Real}) where {d,D,p,q,T}
    uI = ntuple(k -> Distributions.quantile(CC.distortions[k], v[k]), d)
    return _box_partial_cdf(CC.C, CC.is, CC.js, CC.bs, uI, CC.uⱼₛ, CC.lo, CC.hi) / CC.den
end

# Density of the conditional copula on [0,1]^d.
# For v ∈ (0,1)^d, let u_I = quantile(D_k, v_k) with D_k = H_{i_k|J}(·|u_J).
# Then c_{I|J}(v) = f_{U_I|U_J}(u_I|u_J) / ∏_k f_{U_{i_k}|U_J}(u_{i_k}|u_J).
# The Jacobian of v_k = D_k(u_k) contributes 1 / pdf(D_k, u_k).
function Distributions._logpdf(CC::ConditionalCopula{d,D,p,q,T,TDs}, v::AbstractVector{<:Real}) where {d,D,p,q,T,TDs}
    TR = promote_type(eltype(v), T)

    # Support:
    CC.den <= 0 && return TR(-Inf)
    for vₖ in v
        0 < vₖ < 1 || return TR(-Inf)
    end

    # 1) Map v → u_I via the stored distortions (non-sequential conditioning on J only)
    uI = ntuple(k -> Distributions.quantile(CC.distortions[k], v[k]), d)
    # 2)+3) Joint conditional density on the original uniform scale
    if q == 0
        # Full u vector at which to evaluate the base copula density
        u = _assemble(D, CC.is, CC.js, uI, CC.uⱼₛ)
        logdensity = TR(Distributions.logpdf(CC.C, u) - CC.logden)
    else
        # Mixed partial in (J..., I...) of the C-volume over the box on B
        num = _box_partial_cdf(CC.C, (), (CC.js..., CC.is...), CC.bs, (),
                               promote(CC.uⱼₛ..., uI...), CC.lo, CC.hi)
        (num <= 0 || !isfinite(num)) && return TR(-Inf)
        logdensity = TR(log(num) - CC.logden)
    end
    # 4) Change variables from u_I to the conditional marginal scales v
    for idx in 1:d
        logdensity -= Distributions.logpdf(CC.distortions[idx], uI[idx])
    end
    return logdensity
end

# Sampling: sequential inverse-CDF using conditional distortions
function Distributions._rand!(rng::Distributions.AbstractRNG, CC::ConditionalCopula{d, D, p, q, TC}, A::AbstractMatrix{T}) where {T<:Real, d, D, p, q, TC}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    # We want a sample from the COPULA of the conditional model. Let U be a
    # draw from the conditional joint H_{I|J}(· | u_J). The corresponding
    # copula coordinates are V_k = F_{i_k|J}(U_k | u_J) = cdf(distortions[k], U_k).
    # Sample U sequentially by conditioning on J ∪ previously sampled I.
    for col in axes(A, 2)
        J = Int[CC.js...]
        ujs = TC[CC.uⱼₛ...]
        for k in 1:d
            iₖ = CC.is[k]
            uₖ = rand(rng, _distortion_box(CC.C, Tuple(J), Tuple(ujs), CC.bs, CC.lo, CC.hi, iₖ))
            A[k, col] = Distributions.cdf(CC.distortions[k], uₖ)
            push!(J, iₖ)
            push!(ujs, uₖ)
        end
    end
    return A
end

###########################################################################
#####  condition() function
###########################################################################
"""
        condition(C::Copula{D}, js, u_js)
        condition(C::Copula{D}, js, lo_js, hi_js)
        condition(X::SklarDist, js, x_js)
        condition(X::SklarDist, js, xlo_js, xhi_js)

Construct conditional distributions with respect to a copula, either on the
uniform scale (when passing a `Copula`) or on the original data scale (when
passing a `SklarDist`). The three-argument form conditions on the event
`U_js = u_js`; the four-argument form conditions on the box
`U_js ∈ ∏ₖ [lo_js[k], hi_js[k]]`, and a coordinate with `lo == hi` is
conditioned on that point, so `condition(C, js, u, u)` is `condition(C, js, u)`
and one call may mix fixed values with intervals.

Arguments
- `C::Copula{D}`: D-variate copula
- `X::SklarDist`: joint distribution with copula `X.C` and marginals `X.m`
- `js`: indices of conditioned coordinates (tuple, NTuple, or vector)
- `u_js`: values in [0,1] for `U_js` (when conditioning a copula)
- `lo_js, hi_js`: bounds in [0,1], `lo_js ≤ hi_js`, of the box for `U_js`
- `x_js`: values on original scale for `X_js` (when conditioning a SklarDist)
- `xlo_js, xhi_js`: bounds on the original scale of the box for `X_js`
- `j, u_j, x_j`: 1D convenience overloads for the common p = 1 case

Returns
- If the number of remaining coordinates `d = D - length(js)` is 1:
    - `condition(C, js, u_js)` returns the univariate conditional distribution
      of `U_i | U_js = u_js`, supported on `[0,1]`.
    - `condition(X, js, x_js)` returns the univariate conditional distribution
      of `X_i | X_js = x_js` on the original marginal scale.
- If `d > 1`:
    - `condition(C, js, u_js)` returns the conditional joint distribution of
      `U_I | U_js = u_js` on the original copula coordinate scale `[0,1]^d`;
      its conditional margins need not be uniform.
    - `condition(X, js, x_js)` returns the conditional joint distribution on
      the original marginal scales.
- The four-argument forms return the same kinds of objects for the box event.

Notes
- For best performance, pass `js` and `u_js` as NTuple to keep `p = length(js)`
    known at compile time. The specialized method `condition(::Copula{2}, j, u_j)`
    exploits this for the common `D = 2, d = 1` case.
- Specializations are provided for many copula families (Independent, Gaussian, t,
    Archimedean, several bivariate families). Others fall back to an automatic
    differentiation based construction.
- Concrete return types are implementation details. Use the standard
  `Distributions.jl` operations on the returned distribution.

Conditioning on a point is defined through regular conditional laws at the
supplied values. At zero-density points, a generic derivative-based
representation may be undefined or numerically unstable; family specializations
can provide meaningful endpoint or atomic behavior. Conditioning on a box is
defined wherever the box has positive probability, through inclusion–exclusion
of the same CDF partials over its corners; a box of zero probability throws an
`ArgumentError`. Remaining coordinates preserve their original relative order.

Observing a discrete margin `X_j = x_j` of a `SklarDist` is the latent event
`U_j ∈ (F_j(x_j⁻), F_j(x_j)]`, not the point `U_j = F_j(x_j)`
[genest2007](@cite), so `condition(X, js, x_js)` conditions an atom on its
interval and a continuous margin on its point, as the discrete pair-copula
constructions of [panagiotelis2012](@cite) and [schallhorn2017](@cite) do; the
box form `condition(X, js, xlo_js, xhi_js)` maps `[xlo, xhi]` to
`(F_j(xlo⁻), F_j(xhi)]` in the same way.

# Example
```julia
using Copulas, Distributions

C = GaussianCopula(3, 0.4)
D = condition(C, (1,), (0.7,))
cdf(D, [0.4, 0.8])

# U₁ | U₂ = 0.7, U₃ ∈ [0, 0.1]
B = condition(C, (2, 3), (0.7, 0.0), (0.7, 0.1))
cdf(B, 0.4)
```

References:
* [genest2007](@cite) Genest, C., & Nešlehová, J. (2007). A primer on copulas for count data. ASTIN Bulletin, 37(2), 475-515.
* [panagiotelis2012](@cite) Panagiotelis, A., Czado, C., & Joe, H. (2012). Pair copula constructions for multivariate discrete data. Journal of the American Statistical Association, 107(499), 1063-1072.
* [schallhorn2017](@cite) Schallhorn, N., Kraus, D., Nagler, T., & Czado, C. (2017). D-vine quantile regression with discrete variables. arXiv:1705.08310.

See also: [`subsetdims`](@ref), [`rosenblatt`](@ref),
[`inverse_rosenblatt`](@ref), [`SklarDist`](@ref).
"""
function condition(C::Copula{2}, j::Int, uⱼ::Real)
    1 ≤ j ≤ 2 || throw(ArgumentError("Conditioning index must be either 1 or 2."))
    zero(uⱼ) ≤ uⱼ ≤ one(uⱼ) || throw(ArgumentError("Conditioning values must lie in [0, 1]."))
    return distortion(C, (j,), (float(uⱼ),), 3 - j)
end

condition(C::Copula{D}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
# Accept any real `uⱼₛ` (not only `Float64`): `_process_tuples` calls `float.`,
# which keeps `BigFloat`/`Float32` as-is, so a `Float64`-only signature here let
# such inputs fall back to the untyped entry point above and recurse forever
# (StackOverflow). The downstream `DistortionFromCop`/`ConditionalCopula` still
# store `Float64`, so non-`Float64` values are converted there — the conditioning
# result is computed in `Float64` regardless of input precision.
function _conditional_components(C::Copula, js, uⱼₛ, is)
    CC = conditional_copula(C, js, uⱼₛ)
    distortions = CC isa ConditionalCopula ? CC.distortions :
                  Tuple(distortion(C, js, uⱼₛ, i) for i in is)
    return CC, distortions
end

function condition(C::Copula{D}, js::NTuple{p, Int}, uⱼₛ::NTuple{p, <:Real}) where {D, p}
    is = Tuple(setdiff(1:D, js))
    p==D-1 && return distortion(C, js, uⱼₛ, is[1])
    CC, distortions = _conditional_components(C, js, uⱼₛ, is)
    return SklarDist(CC, distortions)
end

# Split the conditioned coordinates into the points (`pt[k]` true) and the box.
function _split_box(js::NTuple{p,Int}, lo::NTuple{p}, hi::NTuple{p}, pt::NTuple{p,Bool}) where {p}
    ps = Tuple(js[k] for k in 1:p if pt[k])
    uₚₛ = Tuple(hi[k] for k in 1:p if pt[k])
    bs = Tuple(js[k] for k in 1:p if !pt[k])
    lob = Tuple(lo[k] for k in 1:p if !pt[k])
    hib = Tuple(hi[k] for k in 1:p if !pt[k])
    return ps, uₚₛ, bs, lob, hib
end

function _condition_box(C::Copula{D}, js::NTuple{p,Int}, lo, hi, pt) where {D,p}
    ps, uₚₛ, bs, lob, hib = _split_box(js, lo, hi, pt)
    isempty(bs) && return condition(C, ps, uₚₛ)
    is = Tuple(i for i in 1:D if i ∉ js)
    length(is) == 1 && return DistortionFromCop(C, ps, uₚₛ, bs, lob, hib, is[1])
    CC = ConditionalCopula(C, ps, uₚₛ, bs, lob, hib)
    return SklarDist(CC, CC.distortions)
end

function condition(C::Copula{D}, js, lo, hi) where {D}
    jst, lot, hit = _process_intervals(Val{D}(), js, lo, hi)
    return _condition_box(C, jst, lot, hit, map(==, lot, hit))
end

# Original-scale conditioning: the latent box on `U_js`, the point coordinates
# being the continuous margins, pushed through the free margins.
function _condition_box(X::SklarDist{<:Copula{D}}, js::NTuple{p,Int}, lo, hi, pt) where {D,p}
    ps, uₚₛ, bs, lob, hib = _split_box(js, lo, hi, pt)
    is = Tuple(i for i in 1:D if i ∉ js)
    if isempty(bs)
        length(is) == 1 && return distortion(X.C, ps, uₚₛ, is[1])(X.m[is[1]])
        CC, distortions = _conditional_components(X.C, ps, uₚₛ, is)
    else
        length(is) == 1 &&
            return DistortionFromCop(X.C, ps, uₚₛ, bs, lob, hib, is[1])(X.m[is[1]])
        CC = ConditionalCopula(X.C, ps, uₚₛ, bs, lob, hib)
        distortions = CC.distortions
    end
    margins = Tuple(distortions[k](X.m[is[k]]) for k in eachindex(is))
    return SklarDist(CC, margins)
end

_is_point_margin(m::Distributions.UnivariateDistribution) =
    Distributions.value_support(typeof(m)) === Distributions.Continuous

condition(C::SklarDist{<:Copula{D}}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
function condition(X::SklarDist{<:Copula{D}, Tpl}, js::NTuple{p, Int}, xⱼₛ::NTuple{p, <:Real}) where {D, Tpl, p}
    bounds = ntuple(k -> _latent_interval(X.m[js[k]], xⱼₛ[k]), p)
    lo = promote(map(first, bounds)...)
    hi = promote(map(last, bounds)...)
    pt = ntuple(k -> _is_point_margin(X.m[js[k]]), p)
    return _condition_box(X, js, lo, hi, pt)
end
function condition(X::SklarDist{<:Copula{D}}, js, xlo, xhi) where {D}
    jst = js isa Integer ? (Int(js),) : Tuple(collect(Int, js))
    xlot = xlo isa Real ? (xlo,) : Tuple(xlo)
    xhit = xhi isa Real ? (xhi,) : Tuple(xhi)
    p = length(jst)
    0 < p < D || throw(ArgumentError("js=$(js) must be a non-empty proper subset of 1:$D"))
    all(in(1:D), jst) && allunique(jst) ||
        throw(ArgumentError("js=$(js) must be distinct indices in 1:$D"))
    length(xlot) == p && length(xhit) == p ||
        throw(ArgumentError("xlo and xhi must have one bound per conditioned coordinate"))
    all(k -> xlot[k] <= xhit[k], 1:p) ||
        throw(ArgumentError("interval bounds must satisfy xlo ≤ xhi"))
    lo = promote(ntuple(k -> first(_latent_interval(X.m[jst[k]], xlot[k])), p)...)
    hi = promote(ntuple(k -> Distributions.cdf(X.m[jst[k]], xhit[k]), p)...)
    pt = ntuple(k -> _is_point_margin(X.m[jst[k]]) && xlot[k] == xhit[k], p)
    return _condition_box(X, jst, lo, hi, pt)
end

###########################################################################
#####  Methods for conditioning subsetcopulas.
###########################################################################


function distortion(S::SubsetCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real}, i::Int) where {p}
    ibase = S.dims[i]
    jsbase = ntuple(k -> S.dims[js[k]], p)
    return distortion(S.C, jsbase, uⱼₛ, ibase)
end

function conditional_copula(S::SubsetCopula{d,CT}, js, uⱼₛ) where {d,CT}
    Jbase = Tuple(S.dims[j] for j in js)
    CC_base = conditional_copula(S.C, Jbase, uⱼₛ)
    D = length(S.C); I = Tuple(setdiff(1:D, Jbase))
    dims_remain = Tuple(i for i in S.dims if !(i in Jbase))
    posmap = Dict(i => p for (p,i) in enumerate(I))
    dims_positions = Tuple(posmap[i] for i in dims_remain)
    return (length(dims_positions) == length(I)) ? CC_base : SubsetCopula(CC_base, dims_positions)
end


###########################################################################
#####  Generic Rosenblatt and inverse Rosenblatt via conditioning
###########################################################################
"""
    rosenblatt(C::Copula, u)
    rosenblatt(X::SklarDist, x)
    rosenblatt([rng::AbstractRNG,] X::SklarDist, x)

Evaluate successive conditional CDFs associated with `C` on the vector `u`.
For `U ∼ C`, the result consists of independent uniforms when the successive
conditional laws are atomless. Forward/inverse round trips hold almost surely
when those CDFs are continuous and invertible on their supports, not universally
for singular or atomic models. Generalized conditional quantiles may still sample
such models through `inverse_rosenblatt`; they do not make the deterministic
forward transform bijective. Matrix inputs evaluate observations columnwise.

Generic Rosenblatt transform using conditional distortions:
S₁ = U₁, S_k = H_{k|1:(k-1)}(U_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.

For a `SklarDist` with continuous margins, `x` is mapped through the marginal
CDFs and the copula transform applies. A discrete margin is an atom: its
observation `x_k` is the latent interval `(F_k(x_k⁻), F_k(x_k)]`, so coordinate
`k` takes the distributional transform [ruschendorf2009](@cite) of its
conditional law, `S_k = H(F_k(x_k⁻)) + V (H(F_k(x_k)) − H(F_k(x_k⁻)))` with
`V ∼ U(0,1)` drawn from `rng`, the randomisation of [brockwell2007](@cite),
and every later coordinate conditions on that interval rather than on the
randomised value, as in the discrete pair-copula constructions of
[panagiotelis2012](@cite). The result is then a vector of independent uniforms
for `X ∼ SklarDist`, and `inverse_rosenblatt` inverts it in law. The rng is
drawn only for atoms; without one, the default rng is used. This is the
convention of vinecopulib's `Vinecop::rosenblatt` with `randomize_discrete`.


* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
* [brockwell2007](@cite) Brockwell, A. E. (2007). Universal residuals: A multivariate transformation. Statistics & Probability Letters, 77(14), 1473-1478.
* [ruschendorf2009](@cite) Rüschendorf, L. (2009). On the distributional transform, Sklar's theorem, and the empirical copula process. Journal of Statistical Planning and Inference, 139(11), 3921-3927.
* [panagiotelis2012](@cite) Panagiotelis, A., Czado, C., & Joe, H. (2012). Pair copula constructions for multivariate discrete data. Journal of the American Statistical Association, 107(499), 1063-1072.

See also: [`inverse_rosenblatt`](@ref), [`condition`](@ref),
[`StatsBase.residuals`](@ref).
"""
rosenblatt(C::Copula{d}, u::AbstractVector{<:Real}) where {d} = rosenblatt(C, reshape(u, (d, 1)))[:]
function rosenblatt(C::Copula{d}, u::AbstractMatrix{<:Real}) where {d}
    size(u, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    v = similar(u)
    @inbounds for j in axes(u, 2)
        # First coordinate is unchanged
        v[1, j] = clamp(float(u[1, j]), 0.0, 1.0)
        for k in 2:d
            js = ntuple(i -> i, k - 1)
            ujs = ntuple(i -> float(u[i, j]), k - 1)  # condition on original u's
            Dk = distortion(C, js, ujs, k)
            v[k, j] = Distributions.cdf(Dk, clamp(float(u[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    _has_atoms(D.m) && return rosenblatt(Random.default_rng(), D, u)
    v = similar(u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.cdf.(Mᵢ, u[i,:])
    end
    return rosenblatt(D.C, v)
end
rosenblatt(D::SklarDist, u::AbstractVector{<:Real}) =
    vec(rosenblatt(D, reshape(u, :, 1)))

# With atoms, coordinate `i` takes the distributional transform of its
# conditional law, `H(F(x⁻)) + V (H(F(x)) − H(F(x⁻)))` with `V ~ U(0,1)`, and
# every successor conditions on the atom's latent interval rather than on the
# randomised latent value: the randomised value is independent of the other
# coordinates, whereas `U_i` on that interval is not. The rng is drawn for
# atoms only; a continuous coordinate is `H(F(x))` as before.
function rosenblatt(rng::Random.AbstractRNG, X::SklarDist{<:Copula{d}}, x::AbstractMatrix{<:Real}) where {d}
    size(x, 1) == d || throw(ArgumentError("Dimension mismatch between distribution and input matrix"))
    _has_atoms(X.m) || return rosenblatt(X, x)
    T = _sklar_work_eltype(X, x)
    S = similar(x, T)
    for col in axes(x, 2)
        ps, uₚₛ, bs, lo, hi = (), (), (), (), ()
        for i in 1:d
            a, b = _latent_interval(X.m[i], x[i, col])
            a, b = clamp(T(a), zero(T), one(T)), clamp(T(b), zero(T), one(T))
            if i == 1
                Ha, Hb = a, b
            else
                Dᵢ = _distortion_box(X.C, ps, uₚₛ, bs, lo, hi, i)
                Ha, Hb = Distributions.cdf(Dᵢ, a), Distributions.cdf(Dᵢ, b)
            end
            S[i, col] = a == b ? Hb : Ha + rand(rng) * (Hb - Ha)
            if a == b
                ps, uₚₛ = (ps..., i), (uₚₛ..., b)
            else
                bs, lo, hi = (bs..., i), (lo..., a), (hi..., b)
            end
        end
    end
    return S
end
rosenblatt(rng::Random.AbstractRNG, X::SklarDist, x::AbstractVector{<:Real}) =
    vec(rosenblatt(rng, X, reshape(x, :, 1)))

"""
    inverse_rosenblatt(C::Copula, u)

Map independent uniform inputs to the dependence structure of `C` by successive
conditional quantiles. If `S` follows the independence copula, the result
follows `C`. Vector inputs represent one point; matrix inputs store points in
columns and are processed without changing their order.

Generic inverse Rosenblatt using conditional distortions:
U₁ = S₁, U_k = H_{k|1:(k-1)}^{-1}(S_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.

Inputs are clamped to the unit interval. Generalized quantiles make the
transformation suitable for sampling conditionals with atoms, but in that case
`rosenblatt(C, inverse_rosenblatt(C, s)) == s` need not hold pointwise. An
almost-sure round trip requires continuous conditional CDFs that are invertible
on their supports.

For a `SklarDist` with discrete margins, step `k` is the generalised inverse
`x_k = Q_k(H⁻¹(s_k))` of the conditional CDF given the predecessors, an atom
among them conditioning on its latent interval [panagiotelis2012](@cite), so
independent uniforms map to the joint law and
`inverse_rosenblatt(X, rosenblatt(rng, X, x)) == x`.


References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
* [panagiotelis2012](@cite) Panagiotelis, A., Czado, C., & Joe, H. (2012). Pair copula constructions for multivariate discrete data. Journal of the American Statistical Association, 107(499), 1063-1072.

See also: [`rosenblatt`](@ref), [`condition`](@ref), [`SklarDist`](@ref).
"""
inverse_rosenblatt(C::Copula{d}, u::AbstractVector{<:Real}) where {d} = inverse_rosenblatt(C, reshape(u, (d, 1)))[:]
function inverse_rosenblatt(C::Copula{d}, s::AbstractMatrix{<:Real}) where {d}
    size(s, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    v = similar(s)
    @inbounds for j in axes(s, 2)
        v[1, j] = clamp(float(s[1, j]), 0.0, 1.0)
        for k in 2:d
            js = ntuple(i -> i, k - 1)
            ujs = ntuple(i -> float(v[i, j]), k - 1)  # use already reconstructed U's
            Dk = distortion(C, js, ujs, k)
            v[k, j] = Distributions.quantile(Dk, clamp(float(s[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function inverse_rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    _has_atoms(D.m) && return _inverse_rosenblatt_atoms(D, u)
    v = inverse_rosenblatt(D.C,u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.quantile.(Mᵢ, v[i,:])
    end
    return v
end

# With atoms, step `i` is the generalised inverse `x_i = Q_i(H⁻¹(s_i))` of the
# conditional CDF given the predecessors, an atom among them conditioning on
# its latent interval. No randomness is needed: the atom that contains `s_i`
# is selected by the margin's quantile.
function _inverse_rosenblatt_atoms(X::SklarDist{<:Copula{d}}, s::AbstractMatrix{<:Real}) where {d}
    size(s, 1) == d || throw(ArgumentError("Dimension mismatch between distribution and input matrix"))
    T = _sklar_work_eltype(X, s)
    x = similar(s, T)
    for col in axes(s, 2)
        ps, uₚₛ, bs, lo, hi = (), (), (), (), ()
        for i in 1:d
            sᵢ = clamp(T(s[i, col]), zero(T), one(T))
            xᵢ = i == 1 ? Distributions.quantile(X.m[1], sᵢ) :
                 Distributions.quantile(_distortion_box(X.C, ps, uₚₛ, bs, lo, hi, i)(X.m[i]), sᵢ)
            x[i, col] = xᵢ
            a, b = _latent_interval(X.m[i], xᵢ)
            a, b = clamp(T(a), zero(T), one(T)), clamp(T(b), zero(T), one(T))
            if a == b
                ps, uₚₛ = (ps..., i), (uₚₛ..., b)
            else
                bs, lo, hi = (bs..., i), (lo..., a), (hi..., b)
            end
        end
    end
    return x
end
inverse_rosenblatt(D::SklarDist, u::AbstractVector{<:Real}) =
    vec(inverse_rosenblatt(D, reshape(u, :, 1)))
