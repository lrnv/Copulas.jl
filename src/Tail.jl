"""
    Tail

Abstract representation of the stable tail dependence function of an
extreme-value copula. A valid STDF `ℓ : [0,∞)^d → [0,∞)` is convex,
one-homogeneous, and satisfies

```math
\\max_i x_i \\leq \\ell(x) \\leq \\sum_i x_i.
```

Equivalently, on the unit simplex it defines a Pickands dependence function
`A(w)=ℓ(w)`. Public component contracts concern documented constructors and
mathematical evaluation. Dimension validation, derivative machinery, density,
conditioning and sampling backends are internal and may require additional
family-specific conditions. See the developer guide for the current architecture.

See also: [`ExtremeValueCopula`](@ref), [`A`](@ref), [`ℓ`](@ref),
[`DiscreteSpectralTail`](@ref).
"""
abstract type Tail end
Base.eltype(tail::Tail) = _sample_eltype(tail)
function (TT::Type{<:Tail})(args...; kwargs...)
    S = hasproperty(TT, :body) ? TT.body : TT
    T = S.name.wrapper
    # Keyword-only calls are deliberately converted to the canonical
    # positional constructor below.  Refuse only a genuinely argument-free
    # recursive call.
    T === TT && isempty(args) && isempty(kwargs) && throw(MethodError(TT, args))
    return T(args..., values(kwargs)...)
end
_parameter_dof(x::Tail) = _parameter_dof(Distributions.params(x))
Base.broadcastable(tail::Tail) = Ref(tail)

"""
    _is_valid_in_dim(tail::Tail, d::Int)

Return whether `tail` defines a valid stable tail dependence function in
dimension `d`. This internal constructor-validation hook defaults to `d ≥ 2`;
bivariate capability types restrict it unless a mathematical family explicitly
provides a multivariate extension.

See also: [`Tail`](@ref), [`BivariatePickandsTail`](@ref), [`ℓ`](@ref).
"""
_is_valid_in_dim(::Tail, d::Int) = d >= 2

"""
    A(tail::Tail, ω)

Evaluate the Pickands representation on the unit simplex. For a generic tail
this is `ℓ(tail, ω)`; bivariate Pickands-capable tails also accept a scalar
coordinate. The input must belong to the representation documented by the
concrete tail.

See also: [`ℓ`](@ref), [`Tail`](@ref), [`ExtremeValueCopula`](@ref).
"""
A(tail::Tail, ω::NTuple{d,<:Real}) where {d} = ℓ(tail, ω)

####### Rest of the interface you can overload if more efficient:
needs_binary_search(::Tail) = false
"""
    ℓ(tail::Tail, x)

Evaluate the stable tail dependence function at a nonnegative vector `x`. The
generic implementation extends `A` from the simplex by one-homogeneity and
returns zero at the origin.

See also: [`A`](@ref), [`Tail`](@ref), [`ExtremeValueCopula`](@ref).
"""
function ℓ(tail::Tail, x)
    s = sum(x)
    return s == 0 ? zero(eltype(x)) : s * A(tail, ntuple(i->x[i]/s, length(x)))
end

"""
    _ellpartial_signlog(tail::Tail, x, I)

Return `(sign, logabs)` for the mixed partial of `ℓ(tail, x)` with respect to
the coordinates in `I`. This internal signed-log protocol avoids overflow and
underflow in extreme-value density and conditioning formulas. The generic
fallback differentiates `ℓ`; specializations must preserve the derivative's
sign and value, with `(0, -Inf)` representing zero.

See also: [`ellpartial`](@ref), [`ℓ`](@ref), [`_mixed_partial`](@ref).
"""
function _ellpartial_signlog(tail::Tail, x, I::Tuple{Vararg{Int}})
    v = _mixed_partial(z -> ℓ(tail, z), x, I)
    iszero(v) && return 0, oftype(v, -Inf)
    return v < zero(v) ? -1 : 1, log(abs(v))
end

_ellpartial_signlog(tail::Tail, x, I::AbstractVector{<:Integer}) = _ellpartial_signlog(tail, x, Tuple(I))

"""
    ellpartial(tail::Tail, x, I)

Evaluate the mixed partial derivative of the STDF with respect to coordinates
`I`. An empty index set returns `ℓ(tail, x)`. This is internal contributor
machinery consumed by generic extreme-value CDF, density and conditioning
algorithms; defining `ℓ` supplies an automatic-differentiation fallback.

See also: [`_ellpartial_signlog`](@ref), [`ℓ`](@ref),
[`ExtremeValueCopula`](@ref).
"""
function ellpartial(tail::Tail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    return iszero(sign) ? zero(logabs) : sign * exp(logabs)
end

ellpartial(tail::Tail, x, I::AbstractVector{<:Integer}) = ellpartial(tail, x, Tuple(I))

"""
    BivariatePickandsTail <: Tail

Internal capability type for tails that provide the scalar bivariate Pickands
function `A(tail, t)`. It activates generic derivatives, density, conditioning
and sampling machinery. The capability is valid only in dimension two by
default; a mathematically valid multivariate family must specialize
`_is_valid_in_dim` explicitly. This subtype is not a stable downstream API.

See also: [`Tail`](@ref), [`A`](@ref), [`dA`](@ref), [`d²A`](@ref),
[`_is_valid_in_dim`](@ref).
"""
abstract type BivariatePickandsTail <: Tail end

# Marker used by fitting routines for one-parameter Pickands families.
abstract type OneParameterPickandsTail <: BivariatePickandsTail end

_is_valid_in_dim(::BivariatePickandsTail, d::Int) = d == 2
A(tail::BivariatePickandsTail, t::NTuple{2, <:Real}) = A(tail, t[1])

"""
    dA(tail::BivariatePickandsTail, t)

Evaluate the first derivative of the scalar Pickands function. The internal
fallback uses forward-mode automatic differentiation; specialized formulas
must retain the same one-sided behavior used by endpoint conditionals.

See also: [`A`](@ref), [`d²A`](@ref), [`BivariatePickandsTail`](@ref).
"""
dA(tail::BivariatePickandsTail, t::Real) = ForwardDiff.derivative(z -> A(tail, z), t)

"""
    d²A(tail::BivariatePickandsTail, t)

Evaluate the second derivative of the scalar Pickands function. The internal
fallback differentiates `dA`. It represents an ordinary derivative and is not
appropriate for atomic spectral curvature, which uses discrete-spectral
machinery instead.

See also: [`A`](@ref), [`dA`](@ref), [`BivariatePickandsTail`](@ref),
[`DiscreteSpectralTail`](@ref).
"""
d²A(tail::BivariatePickandsTail, t::Real) = ForwardDiff.derivative(z -> dA(tail, z), t)

# One-sided Pickands slopes for conditional endpoint extensions.
_pickands_left_slope(tail::BivariatePickandsTail, x::Real) = dA(tail, _safett(zero(x)))
_pickands_right_slope(tail::BivariatePickandsTail, x::Real) = dA(tail, _safett(one(x)))

_A_dA_d²A(tail::BivariatePickandsTail, t::Real) = let tt = _safett(t); (A(tail, tt), dA(tail, tt), d²A(tail, tt)) end
function _biv_der_ℓ(tail::BivariatePickandsTail, uv)
    u, v = uv
    s  = u + v
    x  = u / s
    y  = v / s
    a, da, d2a = _A_dA_d²A(tail, x)
    val  = s * a
    du   = a + da * y
    dv   = a - x * da
    dudv = - x * y * d2a / s
    return val, du, dv, dudv
end
function _ghoudi_mixture_probability(tail::BivariatePickandsTail, z::Real)
    # p(z) = z(1-z) A''(z) / [ A(z) g_Z(z) ] 
    num = z * (1 - z) * d²A(tail, z) 
    dem = A(tail, z) * Distributions.pdf(ExtremeDist(tail), z)
    p = num / dem 
    return clamp(p, 0, 1) 
end

# Finite discrete-spectral capabilities.
#
# `DiscreteSpectralBackedTail` provides a finite spectral representation.
# `DiscreteSpectralPickandsTail` additionally exposes the native scalar
# Pickands interface used by the specialized bivariate algorithms above.
abstract type DiscreteSpectralBackedTail <: Tail end
abstract type DiscreteSpectralPickandsTail <: BivariatePickandsTail end

const DiscreteSpectralCapableTail = Union{
    DiscreteSpectralBackedTail,
    DiscreteSpectralPickandsTail,
}

tail_measure_style(::Tail) = AbsolutelyContinuousMeasure()
@inline limit_kind(::Tail, ::Val) = NO_LIMIT
tail_measure_style(::DiscreteSpectralCapableTail) =
    NonAbsolutelyContinuousMeasure()

# Concrete capable tails store their canonical finite representation in a
# `spectral` field. `DiscreteSpectralTail` itself specializes this accessor.
_spectral_tail(tail::DiscreteSpectralCapableTail) = tail.spectral
ℓ(tail::DiscreteSpectralCapableTail, x) = ℓ(_spectral_tail(tail), x)
