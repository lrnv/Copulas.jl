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
Base.broadcastable(tail::Tail) = Ref(tail)

####### Functions you need to overload: 
_is_valid_in_dim(tail::Tail, d::Int) = throw(ArgumentError("Validity of the tail type $(typeof(tail)) must be supplied by overwriting the function _is_valid_in_dim(tail::Tail, d::Int)"))
A(::Tail, ω::NTuple{d,<:Real}) where {d} = throw(ArgumentError("Implement A(Tail{$d}, ω) en el simplex Δ_{d-1}"))

####### Rest of the interface you can overload if more efficient:
needs_binary_search(::Tail) = false
# \ell function
function ℓ(tail::Tail, x)
    s = sum(x)
    return s == 0 ? zero(eltype(x)) : s * A(tail, ntuple(i->x[i]/s, length(x)))
end


# A more friendly interface for models that are only bivariate: 
abstract type Tail2 <: Tail end
_is_valid_in_dim(::Tail2, d::Int) = (d==2)
A(tail::Tail2, t::NTuple{2, <:Real}) = A(tail, t[1])
dA(tail::Tail2, t::Real) = ForwardDiff.derivative(z -> A(tail, z), t)
d²A(tail::Tail2, t::Real) = ForwardDiff.derivative(z -> dA(tail, z), t)
_A_dA_d²A(tail::Tail2, t::Real) = let tt = _safett(t); (A(tail, tt), dA(tail, tt), d²A(tail, tt)) end
function _biv_der_ℓ(tail::Tail2, uv)
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
function _probability_z(tail::Tail2, z::Real) 
    # p(z) = z(1-z) A''(z) / [ A(z) g_Z(z) ] 
    num = z * (1 - z) * d²A(tail, z) 
    dem = A(tail, z) * _pdf(ExtremeDist(tail), z) # usa pdf, no _pdf 
    p = num / dem 
    return clamp(p, 0, 1) 
end


