"""
    Tail{d}

Abstract type. Implements the API for stable tail dependence functions (STDFs) of extreme-value copulas in dimension `d`.

A STDF is a function
``\\ell : \\mathbb{R}_{+}^d → [0,\\infty)`` that is 1-homogeneous (``\\ell(t·x)=t·\\ell(x)`` for all ``t≥0``), convex, 
and satisfies the bounds
``\\max(x_1,\\ldots,x_d) ≤ \\ell(x) ≤ x_1+ \\cdots +x_d`` (in particular ``\\ell(e_i)=1``).

Pickands representation. By homogeneity, for ``x\\neq 0`` let ``\\left\\| x\\right\\|_1=x_1+\\cdtos+x_d`` and
``\\omega=x/\\left\\| x \\right\\|_1 \\in \\Delta_{d-1}``. There exists a Pickands dependence function
``A:\\Delta_{d-1}\\to [0,1]`` (convex, ``\\max(\\omega_i)≤A(\\omega)≤1``) such that
``\\ell(x)=\\left\\| x\\right\\|_1·A(\\omega)``. For ``d=2``, ``A`` reduces to a convex function on ``[0,1]`` with
``\\max(t,11t)≤A(t)≤1`` and ``A(0)=A(1)=1``.

Interface.
- `A(E::Tail{d}, ω::NTuple{d,Real})` — Pickands function on the simplex `\\Delta_{d-1}`.
  (For `d=2`, a convenience `A(E::Tail{2}, t::Real)` may be provided.)
- `ℓ(E::Tail{d}, x::NTuple{d,Real})` — STDF. By default the package defines
  `ℓ(E, x) = ‖x‖₁ * A(E, x/‖x‖₁)` when `A` is available.

We do not algorithmically verify convexity/bounds; implementers are responsible for validity.

Additional helpers (with defaults).
- For `d=2`: `dA`, `d²A` via AD; stable `logpdf`/`rand` (Ghoudi sampler).
- In any `d`: `cdf(u) = exp(-ℓ(-log.(u)))`.

References:
* Pickands (1981); Gudendorf & Segers (2010); Ghoudi, Khoudraji & Rivest (1998); de Haan & Ferreira (2006).
* Rasell
"""
abstract type Tail{d} end
Base.broadcastable(x::Tail) = Ref(x)
taildim(::Tail{d}) where {d} = d
_is_valid_in_dim(E::Tail, d::Int) = (d == taildim(E))

A(::Tail{d}, ω::NTuple{d,Real}) where {d} = throw(ArgumentError("Implement A(Tail{$d}, ω) en el simplex Δ_{d-1}"))

A(E::Tail{2}, t::Real) = (0.0 ≤ t ≤ 1.0 ? A(E, (t, 1-t)) : throw(ArgumentError("t∈[0,1]")))

function ℓ(E::Tail{d}, x::NTuple{d,Real}) where {d}
    @assert all(>=(0), x) "ℓ requires x ∈ ℝᵈ₊"
    s = sum(x)
    return s == 0 ? zero(eltype(x)) : s * A(E, ntuple(i->x[i]/s, d))
end
