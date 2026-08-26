"""
    Tail

Abstract type. Implements the API for stable tail dependence functions (STDFs) of extreme-value copulas in dimension `d`.

A STDF is a function
``\\ell : \\mathbb{R}_{+}^d → [0,\\infty)`` that is 1-homogeneous (``\\ell(t·x)=t·\\ell(x)`` for all ``t≥0``), convex, 
and satisfies the bounds
``\\max(x_1,\\ldots,x_d) ≤ \\ell(x) ≤ x_1+ \\cdots +x_d`` (in particular ``\\ell(e_i)=1``).

Pickands representation. By homogeneity, for ``x\\neq 0`` let ``\\left\\| x\\right\\|_1=x_1+\\cdots+x_d`` and
``\\omega=x/\\left\\| x \\right\\|_1 \\in \\Delta_{d-1}``. There exists a Pickands dependence function
``A:\\Delta_{d-1}\\to [0,1]`` (convex, ``\\max(\\omega_i)≤A(\\omega)≤1``) such that
``\\ell(x)=\\left\\| x\\right\\|_1·A(\\omega)``. For ``d=2``, ``A`` reduces to a convex function on ``[0,1]`` with
``\\max(t,1-t)≤A(t)≤1`` and ``A(0)=A(1)=1``.

Interface.
- `A(tail::Tail, ω::NTuple{d,Real})` — Pickands function on the simplex `\\Delta_{d-1}`.
  (For `d=2`, a convenience `A(tail::Tail{2}, t::Real)` may be provided.)
- `ℓ(tail::Tail, x::NTuple{d,Real})` — STDF. By default the package defines
  `ℓ(tail, x) = ‖x‖₁ * A(tail, x/‖x‖₁)` when `A` is available.

We do not algorithmically verify convexity/bounds; implementers are responsible for validity.

Additional helpers (with defaults).
- For `d=2`: `dA`, `d²A` via AD; stable `logpdf`/`rand` (Ghoudi sampler).
- In any `d`: `cdf(u) = exp(-ℓ(-log.(u)))`.

References:
* Pickands (1981); Gudendorf & Segers (2010); Ghoudi, Khoudraji & Rivest (1998); de Haan & Ferreira (2006).
* Rasell
"""
abstract type Tail end
function (TT::Type{<:Tail})(args...;kwargs...)
    S = hasproperty(TT, :body) ? TT.body : TT
    T = S.name.wrapper 
    return T(args..., values(kwargs)...)
end
Base.broadcastable(tail::Tail) = Ref(tail)

####### Functions you need to overload: 
_is_valid_in_dim(::Tail, d::Int) = d >= 2
A(::Tail, ω::NTuple{d,<:Real}) where {d} = throw(ArgumentError("Implement A(Tail{$d}, ω) en el simplex Δ_{d-1}"))

####### Rest of the interface you can overload if more efficient:
needs_binary_search(::Tail) = false
# \ell function
function ℓ(tail::Tail, x)
    s = sum(x)
    return s == 0 ? zero(eltype(x)) : s * A(tail, ntuple(i->x[i]/s, length(x)))
end

# Mixed STDF partials. A new Tail only needs to implement ℓ.
# Generic mixed partials come from the shared AD helper.
function _ellpartial_signlog(tail::Tail, x, I::Tuple{Vararg{Int}})
    v = _mixed_partial(z -> ℓ(tail, z), x, I)
    iszero(v) && return 0, oftype(v, -Inf)
    return v < zero(v) ? -1 : 1, log(abs(v))
end

_ellpartial_signlog(tail::Tail, x, I::AbstractVector{<:Integer}) = _ellpartial_signlog(tail, x, Tuple(I))

function ellpartial(tail::Tail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    return iszero(sign) ? zero(logabs) : sign * exp(logabs)
end

ellpartial(tail::Tail, x, I::AbstractVector{<:Integer}) = ellpartial(tail, x, Tuple(I))

# Native scalar Pickands interface in d=2.
#
# `BivariatePickandsTail` is a computational capability: the tail provides the
# scalar Pickands representation A(t) and therefore has access to the mature
# bivariate derivative, density, conditioning, and sampling machinery.
#
# The capability is bivariate by default. Mathematical families that also have
# a valid multivariate STDF override `_is_valid_in_dim`.
abstract type BivariatePickandsTail <: Tail end

# Marker used by fitting routines for one-parameter Pickands families.
abstract type OneParameterPickandsTail <: BivariatePickandsTail end

_is_valid_in_dim(::BivariatePickandsTail, d::Int) = d == 2
A(tail::BivariatePickandsTail, t::NTuple{2, <:Real}) = A(tail, t[1])
dA(tail::BivariatePickandsTail, t::Real) = ForwardDiff.derivative(z -> A(tail, z), t)
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

# Concrete capable tails store their canonical finite representation in a
# `spectral` field. `DiscreteSpectralTail` itself specializes this accessor.
_spectral_tail(tail::DiscreteSpectralCapableTail) = tail.spectral
ℓ(tail::DiscreteSpectralCapableTail, x) = ℓ(_spectral_tail(tail), x)
