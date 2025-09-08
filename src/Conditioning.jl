###########################################################################
#####  Conditioning framework: Distortion, DistortionFromCop, 
#####                          DistortedDist, ConditionalCopula
#####  User-facing function: `condition(), ropsenblatt(), inverse_rosenblatt()`
###########################################################################

# usefull functions: 
function _assemble(D, is, js, uᵢₛ, uⱼₛ)
    Tᵢ = eltype(typeof(uᵢₛ)); Tⱼ = eltype(typeof(uⱼₛ)); T = promote_type(Tᵢ, Tⱼ)
    w = fill(one(T), D)
    @inbounds for (k,i) in pairs(is); w[i] = uᵢₛ[k]; end
    @inbounds for (k,j) in pairs(js); w[j] = uⱼₛ[k]; end
    return w
end
function _swap(u, i, uᵢ)
    T = promote_type(eltype(u), typeof(uᵢ))
    v = similar(u, T)
    @inbounds for k in eachindex(u)
        v[k] = u[k]
    end
    v[i] = uᵢ
    return v
end
_der(f, u, i::Int) = ForwardDiff.derivative(uᵢ -> f(_swap(u, i, uᵢ)), u[i])
_der(f, u, is::NTuple{1,Int}) = _der(f, u, is[1])
_der(f, u, is::NTuple{N,Int}) where {N} = _der(u′ -> _der(f, u′, (is[end],)), u, is[1:end-1])
_partial_cdf(C, is, js, uᵢₛ, uⱼₛ) = _der(u -> Distributions.cdf(C, u), _assemble(length(C), is, js, uᵢₛ, uⱼₛ), js)

_process_tuples(::Val{D}, js::NTuple{p, Int64}, ujs::NTuple{p, Float64}) where {D,p} = (js, ujs) 
_process_tuples(::Val{D}, j::Int64, uj::Real) where {D} = ((j,), (uj,)) 
function _process_tuples(::Val{D}, js, ujs) where D
    p, p2 = length(js), length(ujs)
    @assert 0 < p < D "js=$(js) must be a non-empty proper subset of 1:D of length at most D-1 (D = $D)"
    @assert p == p2 && all(0 .<= ujs .<= 1) "uⱼₛ must be in [0,1] and match js length"
    jst = Tuple(collect(Int, js))
    @assert all(in(1:D), jst)
    ujst = Tuple(collect(float.(ujs)))
    return (jst, ujst)
end

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
function Distributions.quantile(d::Distortion, α::Real) 
    T = typeof(float(α))
    ϵ = eps(T)
    α < ϵ && return zero(T)
    α > 1 - 2ϵ && return one(T)
    lα = log(α)
    f(u) = Distributions.logcdf(d, u) - lα
    return Roots.find_zero(f, (ϵ, 1 - 2ϵ), Roots.Bisection(); xtol = sqrt(eps(T)))
end
# You have to implement one of these two: 
Distributions.logcdf(d::Distortion, t::Real) = log(Distributions.cdf(d, t))
Distributions.cdf(d::Distortion, t::Real) = exp(Distributions.logcdf(d, t))

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
    i::Int
    js::NTuple{p,Int}
    uⱼₛ::NTuple{p,Float64}
    den::Float64
    function DistortionFromCop(C::Copula{D}, js, uⱼₛ, i) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        p = length(jst)
        if p==1
            den = Distributions.pdf(subsetdims(C, jst), uⱼₛt[1])
        else
            den = Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        end
        return new{typeof(C), p}(C, i, jst, uⱼₛt, den)
    end
end
Distributions.cdf(d::DistortionFromCop, u::Real) = _partial_cdf(d.C, (d.i,), d.js, (u,), d.uⱼₛ) / d.den

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
struct ConditionalCopula{d, D, p, TDs}<:Copula{d}
    C::Copula{D}
    js::NTuple{p, Int}
    is::NTuple{d, Int}
    uⱼₛ::NTuple{p, Float64}
    den::Float64
    distortions::TDs
    function ConditionalCopula(C::Copula{D}, js, uⱼₛ) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        ist = Tuple(setdiff(1:D, jst))
        p = length(jst)
        d = D - p
        distos = Tuple(DistortionFromCop(C, jst, uⱼₛt, i) for i in ist)
                if p==1
            den = Distributions.pdf(subsetdims(C, jst), uⱼₛt[1])
        else
            den = Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        end
        return new{d, D, p, typeof(distos)}(C, jst, ist, uⱼₛt, den, distos)
    end
end
function _cdf(CC::ConditionalCopula{d,D,p,T}, v::AbstractVector{<:Real}) where {d,D,p,T}
    return _partial_cdf(CC.C, CC.is, CC.js, Distributions.quantile.(CC.distortions, v), CC.uⱼₛ) / CC.den
end

###########################################################################
#####  condition() function
###########################################################################
"""
        condition(C::Copula{D}, js, u_js)
        condition(X::SklarDist, js, x_js)

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
condition(C::Copula{D}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
function condition(C::Copula{D}, js::NTuple{p, Int}, uⱼₛ::NTuple{p, Float64}) where {D, p}
    margins = Tuple(DistortionFromCop(C, js, uⱼₛ, i) for i in setdiff(1:D, js))
    p==D-1 && return margins[1]
    return SklarDist(ConditionalCopula(C, js, uⱼₛ), margins)
end

condition(C::SklarDist{<:Copula{D}}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
function condition(X::SklarDist{<:Copula{D}, Tpl}, js::NTuple{p, Int}, xⱼₛ::NTuple{p, Float64}) where {D, Tpl, p}
    uⱼₛ = Tuple(Distributions.cdf(X.m[j], xⱼ) for (j,xⱼ) in zip(js, xⱼₛ))
    margins = Tuple(DistortionFromCop(X.C, js, uⱼₛ, i)(X.m[i]) for i in setdiff(1:D, js))
    p==D-1 && return margins[1]
    return SklarDist(ConditionalCopula(X.C, js, uⱼₛ), margins)
end

###########################################################################
#####  Methods for conditioning subsetcopulas. 
###########################################################################


@inline function DistortionFromCop(S::SubsetCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p}
    ibase = S.dims[i]
    jsbase = ntuple(k -> S.dims[js[k]], p)
    return DistortionFromCop(S.C, jsbase, uⱼₛ, ibase)
end

function ConditionalCopula(S::SubsetCopula{d,CT}, js, uⱼₛ) where {d,CT}
    Jbase = Tuple(S.dims[j] for j in js)
    CC_base = ConditionalCopula(S.C, Jbase, uⱼₛ)
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

Computes the rosenblatt transform associated to the copula C on the vector u. Formally, assuming that U ∼ C, the result should be uniformely distributed on the unit hypercube. The importance of this transofrmation comes from its bijectivity: `inverse_rosenblatt(C, rand(d))` is equivalent to `rand(C)`. The interface proposes faster versions for matrix inputs `u`.

Generic Rosenblatt transform using conditional distortions:
S₁ = U₁, S_k = H_{k|1:(k-1)}(U_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.


* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
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
            Dk = DistortionFromCop(C, js, ujs, k)
            v[k, j] = Distributions.cdf(Dk, clamp(float(u[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    v = similar(u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.cdf.(Mᵢ, u[i,:])
    end
    return rosenblatt(D.C, v)
end

"""
    inverse_rosenblatt(C::Copula, u)

Computes the inverse rosenblatt transform associated to the copula C on the vector u. Formally, assuming that U ∼ Π, the independence copula, the result should be distributed as C. Also look at `rosenblatt(C, u)` for the inverse transformation. The interface proposes faster versions for matrix inputs `u`. 

Generic inverse Rosenblatt using conditional distortions:
U₁ = S₁, U_k = H_{k|1:(k-1)}^{-1}(S_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.


References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
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
            Dk = DistortionFromCop(C, js, ujs, k)
            v[k, j] = Distributions.quantile(Dk, clamp(float(s[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function inverse_rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    v = inverse_rosenblatt(D.C,u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.quantile.(Mᵢ, v[i,:])
    end
    return v
end

