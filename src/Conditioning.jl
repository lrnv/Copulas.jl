###########################################################################
#####  Conditioning framework: Distortion, DistortionFromCop, DistortedDist, ConditionalCopula, condition
###########################################################################

"""
    Distortion <: Distributions.ContinuousUnivariateDistribution

Abstract super-type for objects describing the (uniform-scale) conditional marginal
transformation U_i | U_J = u_J of a copula.

Subtypes implement cdf/quantile on [0,1]. They are not full arbitrary distributions;
they model how a uniform variable is distorted by conditioning. They can be applied
as a function to a base marginal distribution to obtain the conditional marginal on
the original scale: if `D::Distortion` and `X::UnivariateDistribution`, then `D(X)`
is the distribution of `X_i | U_J = u_J`.
"""
abstract type Distortion<:Distributions.ContinuousUnivariateDistribution end
(D::Distortion)(::Distributions.Uniform) = D
(D::Distortion)(X::Distributions.UnivariateDistribution) = DistortedDist(D, X)
Distributions.minimum(::Distortion) = 0.0
Distributions.maximum(::Distortion) = 1.0

"""
        DistortionFromCop{TC,p,T} <: Distortion

Generic, uniform-scale conditional marginal transformation for a copula.

This is the default fallback (based on mixed partial derivatives computed via
automatic differentiation) used when a faster specialized `Distortion` is not
available for a given copula family.

Parameters
- `TC`: copula type
- `p`: length of the conditioned index set J (static)
- `T`: element type for the conditioned values u_J

Construction
- `DistortionFromCop(C::Copula, js::NTuple{p,Int}, ujs::NTuple{p,<:Real}, i::Int)`
    builds the distortion for the conditional marginal of index `i` given `U_js = ujs`.

Notes
- A convenience method `DistortionFromCop(C, j::Int, uj::Real, i::Int)` exists for
    the common `p = 1` case.
"""
struct DistortionFromCop{TC,p}<:Distortion
    C::TC
    i::Int64
    js::NTuple{p,Int64}
    uⱼₛ::NTuple{p,Float64}
    den::Float64
    function DistortionFromCop(C::Copula{D}, js::NTuple{p,Int64}, uⱼₛ::NTuple{p,T}, i::Int64) where {D,p,T}
        den = p==1 ? Distributions.pdf(subsetdims(C, js), uⱼₛ[1]) : Distributions.pdf(subsetdims(C, js), collect(uⱼₛ))
        return new{typeof(C), p}(C, i, js, float.(uⱼₛ), den)
    end
end
@inline DistortionFromCop(C::Copula{D}, j::Int64, uⱼ::Real, i::Int64) where {D} = DistortionFromCop(C, (Int(j),), (float(uⱼ),), i)
Distributions.cdf(d::DistortionFromCop, u::Real) = _∂C_∂uⱼₛ(d.C, (d.i,), d.js, (u,), d.uⱼₛ) / d.den
function Distributions.quantile(d::DistortionFromCop, α::Real) 
    T = typeof(float(α))
    return Roots.find_zero(u -> T(α) - cdf(d, u), (zero(T), one(T)), Roots.Bisection(); xtol=sqrt(eps(T)))
end

@inline function _assemble(D::Int, is, js, uᵢₛ, uⱼₛ)
    Tᵢ = eltype(typeof(uᵢₛ)); Tⱼ = eltype(typeof(uⱼₛ)); T = promote_type(Tᵢ, Tⱼ)
    w = fill(one(T), D)
    @inbounds for (k,i) in pairs(is); w[i] = uᵢₛ[k]; end
    @inbounds for (k,j) in pairs(js); w[j] = uⱼₛ[k]; end
    return w
end
@inline function _swap(u, i, uᵢ)
    T = promote_type(eltype(u), typeof(uᵢ))
    v = similar(u, T); @inbounds for k in eachindex(u); v[k] = u[k]; end; v[i] = uᵢ; return v
end
@inline _der(f::FT, u, i::Int) where FT = ForwardDiff.derivative(uᵢ -> f(_swap(u, i, uᵢ)), u[i])
@inline _der(f::FT, u, is::NTuple{1,Int}) where FT = _der(f::FT, u, is[1])
@inline _der(f::FT, u, is::NTuple{N,Int}) where {N, FT} = _der(u′ -> _der(f, u′, (is[end],)), u, is[1:end-1])
@inline _∂C_∂uⱼₛ(C, is, js, uᵢₛ, uⱼₛ) = _der(u -> Distributions.cdf(C, u), _assemble(length(C), is, js, uᵢₛ, uⱼₛ), js)

"""
    DistortedDist{Disto,Distrib} <: Distributions.UnivariateDistribution

Push-forward of a base marginal by a `Distortion`.
"""
struct DistortedDist{Disto, Distrib}<:Distributions.ContinuousUnivariateDistribution
    D::Disto
    X::Distrib
    function DistortedDist(D::Distortion, X::Distributions.UnivariateDistribution)
        return new{typeof(D), typeof(X)}(D, X)
    end
end
Distributions.cdf(D::DistortedDist, t::Real) = Distributions.cdf(D.D, Distributions.cdf(D.X, t))
Distributions.quantile(D::DistortedDist, α::Real) = Distributions.quantile(D.X, Distributions.quantile(D.D, α))

"""
    ConditionalCopula{d} <: Copula{d}

Copula of the conditioned random vector U_I | U_J = u_J.
"""
struct ConditionalCopula{d,D,p, TDs}<:Copula{d}
    C::Copula{D}
    js::NTuple{p, Int}
    is::NTuple{d, Int}
    uⱼₛ::NTuple{p, Float64}
    den::Float64
    distortions::TDs
    function ConditionalCopula(C::Copula{D}, js, uⱼₛ) where {D}
        p, p2 = length(js), length(uⱼₛ)
        d = D - p
        @assert 0 < p < D-1 "js=$(js) must be a non-empty proper subset of 1:D of length at most D-2 (D = $D)"
        @assert p == p2 && all(0 .<= uⱼₛ .<= 1) "uⱼₛ must be in [0,1] and match js length"
        jst = Tuple(collect(Int, js))
        @assert all(in(1:D), jst)
        ist = Tuple(setdiff(1:D, jst))
        uⱼₛt = Tuple(collect(float.(uⱼₛ)))
        distos = Tuple(DistortionFromCop(C, jst, uⱼₛt, i) for i in ist)
        den = p==1 ? Distributions.pdf(subsetdims(C, jst), uⱼₛ[1]) : Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        return new{d, D, p, typeof(distos)}(C, jst, ist, uⱼₛt, den, distos)
    end
end
function _cdf(CC::ConditionalCopula{d,D,p,T}, v::AbstractVector{<:Real}) where {d,D,p,T}
    u = Distributions.quantile.(CC.distortions, v)
    return _∂C_∂uⱼₛ(CC.C, CC.is, CC.js, u, CC.uⱼₛ) / CC.den
end

###########################################################################
#####  condition() function
###########################################################################
"""
        condition(C::Copula{D}, js, u_js)
        condition(C::Copula{2}, j::Int, u_j::Real)
        condition(X::SklarDist, js, x_js)
        condition(X::SklarDist, j::Int, x_j::Real)

Construct conditional distributions with respect to a copula, either on the
uniform scale (when passing a `Copula`) or on the original data scale (when
passing a `SklarDist`).

Arguments
- `C::Copula{D}`: D-variate copula
- `X::SklarDist`: joint distribution with copula `X.C` and marginals `X.m`
- `js`: indices of conditioned coordinates (tuple, NTuple, or vector)
- `u_js`: values in [0,1] for `U_js` (when conditioning a copula)
- `x_js`: values on original scale for `X_js` (when conditioning a SklarDist)
- `j, u_j, x_j`: 1D convenience overloads for the common p = 1 case

Returns
- If the number of remaining coordinates `d = D - length(js)` is 1:
    - `condition(C, js, u_js)` returns a `Distortion` on [0,1] describing
        `U_i | U_js = u_js`.
    - `condition(X, js, x_js)` returns an unconditional univariate distribution
        for `X_i | X_js = x_js`, computed as the push-forward `D(X.m[i])` where
        `D = condition(C, js, u_js)` and `u_js = cdf.(X.m[js], x_js)`.
- If `d > 1`:
    - `condition(C, js, u_js)` returns the conditional joint distribution on
        the uniform scale as a `SklarDist(ConditionalCopula, distortions)`.
    - `condition(X, js, x_js)` returns the conditional joint distribution on the
        original scale as a `SklarDist` with copula `ConditionalCopula(C, js, u_js)` and
        appropriately distorted marginals `D_k(X.m[i_k])`.

Notes
- For best performance, pass `js` and `u_js` as NTuple to keep `p = length(js)`
    known at compile time. The specialized method `condition(::Copula{2}, j, u_j)`
    exploits this for the common `D = 2, d = 1` case.
- Specializations are provided for many copula families (Independent, Gaussian, t,
    Archimedean, several bivariate families). Others fall back to an automatic
    differentiation based construction.
- This function returns the conditional joint distribution `H_{I|J}(· | u_J)`.
    The “conditional copula” is `ConditionalCopula(C, js, u_js)`, i.e., the copula
    of that conditional distribution.
"""
function condition(C::Copula{D}, js, uⱼₛ) where {D}
    is = Tuple(setdiff(1:D, js))
    d = length(is)
    if d==1
        jst = isa(js, NTuple) ? js : Tuple(collect(Int, js))
        ujt = isa(uⱼₛ, NTuple) ? uⱼₛ : Tuple(collect(float.(uⱼₛ)))
        return DistortionFromCop(C, jst, ujt, is[1])
    else
        CCond = ConditionalCopula(C, js, uⱼₛ)
        jst = isa(js, NTuple) ? js : Tuple(collect(Int, js))
        ujt = isa(uⱼₛ, NTuple) ? uⱼₛ : Tuple(collect(float.(uⱼₛ)))
        marg = Tuple(DistortionFromCop(C, jst, ujt, i) for i in is)
        return SklarDist(CCond, marg)
    end
end
@inline function condition(C::Copula{2}, j::Int, uⱼ::Real)
    @assert j==1 || j==2
    i = (j==1) ? 2 : 1
    return DistortionFromCop(C, (Int(j),), (float(uⱼ),), i)
end
function condition(X::SklarDist, js, xⱼₛ)
    D = length(X)
    jst = Tuple(collect(Int, js))
    ist = Tuple(setdiff(1:D, jst))
    d = length(ist)
    uⱼₛ = Tuple(Distributions.cdf(X.m[j], xⱼ) for (j,xⱼ) in zip(jst, xⱼₛ))
    if d==1
        return DistortionFromCop(X.C, jst, uⱼₛ, ist[1])(X.m[ist[1]])
    else
        Ccond = ConditionalCopula(X.C, jst, uⱼₛ)
        marg = Tuple(DistortionFromCop(X.C, jst, uⱼₛ, i)(X.m[i]) for i in ist)
        return SklarDist(Ccond, marg)
    end
end
@inline condition(CX::T, j::Int, xⱼ::Real) where T<:Union{Copula, SklarDist} = condition(CX, (Int(j),), (float(xⱼ),))

