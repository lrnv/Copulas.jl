"""
    NestedArchimedeanCopula{d, TG}

A nested (hierarchical) Archimedean copula: an outer Archimedean
[`Generator`](@ref) acting on a mix of bare coordinates and inner Archimedean
sub-copulas, each occupying its own block of dimensions. With a single outer
generator over inner copulas ``C_1, \\dots, C_m`` on disjoint coordinate blocks
``I_1, \\dots, I_m`` (and possibly some bare coordinates attached directly to the
root), the CDF is

```math
C(\\mathbf u) = \\phi_0\\!\\left(
  \\sum_{i \\in \\text{root leaves}} \\phi_0^{-1}(u_i)
  \\;+\\; \\sum_{k=1}^m \\phi_0^{-1}\\bigl(C_k(\\mathbf u_{I_k})\\bigr)
\\right),
```

where ``\\phi_0`` is the outer generator and each child ``C_k`` is itself an
Archimedean (or nested Archimedean) copula. The construction nests to arbitrary
depth.

# Constructor

    NestedArchimedeanCopula(G::Generator; leaves = Int[], children = [])

* `G`        — the outer generator ``\\phi_0``.
* `leaves`   — dimension indices of bare coordinates attached directly to `G`.
* `children` — inner sub-copulas. Each entry is either
    - a sub-copula (an [`ArchimedeanCopula`](@ref) or a `NestedArchimedeanCopula`,
      so trees nest to arbitrary depth), which is auto-placed on the next free
      block of dimensions in declaration order, or
    - a `Pair` `sub_copula => dims`, which places the child on the explicit
      dimension indices `dims`.

The bare `leaves` and the children's dimension blocks must together tile
`1:d` exactly (no gaps, no overlaps).

A purely flat declaration (only `leaves`, no `children`) returns the package's
native [`ArchimedeanCopula`](@ref) so its fast specialised density is used; only
genuinely nested declarations build a `NestedArchimedeanCopula`. The legacy
positional form `NestedArchimedeanCopula(G, children)` (children in consecutive
blocks, no root leaves) is also supported.

The constructor does **not** check the nesting condition: the caller is
responsible for supplying a parameter combination for which the nested
construction is a valid copula (for same-family nestings this means the inner
generator is at least as dependent as the outer one).

# Density and precision

The (log-)density is computed via Faà di Bruno's formula / partial Bell
polynomials over the generator tree. The recursion is generic in the value type:
`logpdf` on `Float64` coordinates works out of the box, while passing `BigFloat`
(or `Double64`) coordinates carries that precision through the whole recursion —
recommended for adversarial high-dimensional or deep-tail inputs where the
alternating-sign Faà di Bruno sum can lose `Float64` precision.

# Censored / survival likelihood

Per-variable (right-)censoring is an *emergent* capability of the standard
[`condition`](@ref) + [`subsetdims`](@ref) framework — there is no bespoke
censored-likelihood function. For an observed set ``O`` and a censored set ``C``,
the per-variable censored likelihood factorises as the "gist recipe"

```julia
logpdf(subsetdims(X, O), x_O) + logcdf(condition(X, O, x_O), x_C)
```

(on the [`SklarDist`](@ref) `X = SklarDist(C, margins)`), which equals the
observed-marginal densities times the mixed partial of the nested CDF over the
observed coordinates. The `subsetdims`/`condition` specialisations for this type
route both factors through the Faà di Bruno tree walk; the denominator
``c_O`` cancels, reproducing the raw mixed partial.

# Example

```julia
using Copulas: ClaytonGenerator   # generator types are not exported

# outer Clayton(2) over two inner Clayton panels on dims 1:2 and 3:4
C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
        children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
logpdf(C, [0.3, 0.5, 0.4, 0.6])

# survival likelihood with dim 2 right-censored (observed O = {1,3,4}, C = {2}):
S = SklarDist(C, ntuple(_ -> Exponential(1.0), 4))
x = [0.7, 0.3, 0.5, 0.9]
logpdf(subsetdims(S, (1, 3, 4)), x[[1, 3, 4]]) +
    logcdf(condition(S, (1, 3, 4), x[[1, 3, 4]]), x[2])
```

The density and the per-variable censored (survival) likelihood follow the
algorithm of Yang & Li (arXiv:2605.23134), computed via Faà di Bruno's formula /
partial Bell polynomials over the generator tree.

References:
* Yang, C. & Li, D. "Archimedean Copula Inference via Taylor-Mode AD."
  arXiv:2605.23134 (2026). https://arxiv.org/abs/2605.23134
"""
struct NestedArchimedeanCopula{d, TG<:Generator} <: Copula{d}
    G::TG
    leafdims::Vector{Int}     # dims of bare leaves attached directly to G
    children::Vector{Any}     # each: (ArchimedeanCopula, dims) | NestedArchimedeanCopula
    dims::Vector{Int}         # all dims covered by this subtree (sorted)
end

Base.length(::NestedArchimedeanCopula{d}) where {d} = d

# Element type of a single generator's parameters (promote across its params).
# `init = Bool` is the identity for `promote_type`, so a 0-param generator
# yields `Bool` and never widens the data type.
_gen_param_eltype(G::Generator) =
    mapreduce(typeof, promote_type, values(Distributions.params(G)); init = Bool)

# Promote the parameter element type over the WHOLE tree (root + every child /
# nested node). Used to widen the Faà di Bruno working type `T` so that
# generator params of type `ForwardDiff.Dual` (e.g. pushed by an optimizer
# through Float64 data) flow through the recursion alongside the leaves. For
# plain `Float64` data + params this returns `Float64` (a no-op).
function _tree_param_eltype(C::NestedArchimedeanCopula)
    T = _gen_param_eltype(C.G)
    for ch in C.children
        if ch isa Tuple                       # (flat ArchimedeanCopula, dims)
            T = promote_type(T, _gen_param_eltype(ch[1].G))
        else                                  # nested child
            T = promote_type(T, _tree_param_eltype(ch))
        end
    end
    return T
end

# Dimension count of a sub-copula entry.
_subdim(c::ArchimedeanCopula) = length(c)
_subdim(c::NestedArchimedeanCopula{d}) where {d} = d
_subdim(c::Tuple) = _subdim(c[1])

# ---- Unified keyword constructor --------------------------------------------
function NestedArchimedeanCopula(G::Generator;
                                 leaves::AbstractVector{<:Integer} = Int[],
                                 children::AbstractVector = Any[])
    leafdims = collect(Int, leaves)
    kids = Any[]
    kiddims = Vector{Int}[]
    used = Set{Int}(leafdims)
    length(leafdims) == length(used) || throw(ArgumentError("duplicate dimension in `leaves`: $(leafdims)"))

    pending = Any[]  # children awaiting automatic dimension assignment
    for ch in children
        if ch isa Pair
            c, ds = ch.first, collect(Int, ch.second)
            length(ds) == _subdim(c) || throw(ArgumentError("child block $(ds) has size ≠ child dimension $(_subdim(c))"))
            for x in ds
                (x in used) && throw(ArgumentError("dimension $x assigned to more than one child/leaf"))
                push!(used, x)
            end
            push!(kids, c); push!(kiddims, ds)
        else
            push!(pending, ch)
        end
    end
    # Auto-assign pending children to the lowest free dims, as consecutive blocks
    # in declaration order (this reproduces the legacy positional behaviour).
    nextfree() = (i = 1; while i in used; i += 1; end; i)
    for c in pending
        k = _subdim(c); ds = Int[]; s = nextfree()
        for j in 0:k-1
            push!(ds, s + j); push!(used, s + j)
        end
        push!(kids, c); push!(kiddims, ds)
    end

    alldims = sort(collect(used))
    d = length(alldims)
    alldims == collect(1:d) || throw(ArgumentError("declared dimensions $(alldims) must tile 1:$d with no gaps or overlaps"))

    # Flat declaration (no children) → native ArchimedeanCopula (fast path).
    isempty(kids) && return ArchimedeanCopula(length(leafdims), G)

    kids2 = Any[_place_dims(kids[i], kiddims[i]) for i in eachindex(kids)]
    return NestedArchimedeanCopula{d, typeof(G)}(G, leafdims, kids2, alldims)
end

# Legacy positional form: children in consecutive blocks, no root leaves.
NestedArchimedeanCopula(G::Generator, children::AbstractVector) =
    NestedArchimedeanCopula(G; leaves = Int[], children = collect(Any, children))

# ---- Dimension placement ----------------------------------------------------
# A flat child keeps its generator and is tagged with its (global) dims.
_place_dims(c::ArchimedeanCopula, ds::Vector{Int}) = (c, ds)
function _place_dims(c::NestedArchimedeanCopula, ds::Vector{Int})
    length(ds) == length(c.dims) || throw(ArgumentError("nested child block size mismatch"))
    remap = Dict(c.dims[i] => ds[i] for i in eachindex(c.dims))
    return _remap_dims(c, remap)
end
function _remap_dims(c::NestedArchimedeanCopula, remap::Dict{Int,Int})
    newleaf = [remap[x] for x in c.leafdims]
    newkids = Any[]
    for ch in c.children
        if ch isa Tuple
            cc, ds = ch
            push!(newkids, (cc, [remap[x] for x in ds]))
        else
            push!(newkids, _remap_dims(ch, remap))
        end
    end
    newdims = sort([remap[x] for x in c.dims])
    return NestedArchimedeanCopula{length(newdims), typeof(c.G)}(c.G, newleaf, newkids, newdims)
end

# ---- Build the internal density tree from structure + data + censoring ------
function _build_tree(C::NestedArchimedeanCopula, u, censored, ::Type{T}) where {T}
    lvs = [_NestedLeaf{T}(T(u[i]), censored[i]) for i in C.leafdims]
    kids = _NestedNode[]
    for ch in C.children
        if ch isa Tuple                       # (flat ArchimedeanCopula, dims)
            cc, ds = ch
            clvs = [_NestedLeaf{T}(T(u[i]), censored[i]) for i in ds]
            push!(kids, _NestedNode(cc.G, clvs, _NestedNode[]))
        else                                  # nested child
            push!(kids, _build_tree(ch, u, censored, T))
        end
    end
    return _NestedNode(C.G, lvs, kids)
end

# ---- Distributions.jl interface ---------------------------------------------
# Uncensored nested log-density (the standard Distributions._logpdf entry point).
function Distributions._logpdf(C::NestedArchimedeanCopula{d}, u) where {d}
    if !all(0 .< u .< 1)
        return eltype(u)(-Inf)
    end
    Tu = eltype(u) <: AbstractFloat ? float(eltype(u)) : Float64
    # Promote with the generator-param eltype of the whole tree so that
    # `Dual`-typed generator params (e.g. from an optimizer differentiating wrt
    # θ through Float64 data) flow through the Faà di Bruno recursion together
    # with the leaves. No-op (`Float64`) for plain Float64 data + params.
    T = promote_type(Tu, _tree_param_eltype(C))
    tree = _build_tree(C, u, falses(d), T)
    return _nested_logpdf(tree)
end

# Nested CDF in closed form (ϕ_root ∘ Σ ϕ⁻¹), avoiding the generic numerical
# integration fallback. This is the joint CDF used by `rosenblatt` and is also
# the all-censored limit of the survival likelihood (the empty-observed gist
# recipe `log cdf(C, u)`). NOTE: the MULTI-censored-dim `ConditionalCopula`
# conditional CDF is NO LONGER obtained by ForwardDiff-differentiating this
# closed form — it is computed directly by our Faà di Bruno kernel via the
# `_partial_cdf(::NestedArchimedeanCopula, …)` override in
# nested/NestedConditioning.jl. The `eltype`-preserving `T` below still lets
# ForwardDiff `Dual`s flow through for any residual generic differentiation of
# the CDF (e.g. non-nested callers), but it is not the multi-censored path.
function _cdf(C::NestedArchimedeanCopula{d}, u) where {d}
    # Preserve the input element type when it is a (non-integer) real so that
    # ForwardDiff `Dual`s flow through unchanged for any generic caller.
    Tu = eltype(u)
    T = Tu <: Integer ? Float64 : (Tu <: Real ? float(Tu) : Float64)
    tree = _build_tree(C, u, falses(d), T)
    return _nested_cdf(tree)
end

# Sampling. NestedArchimedeanCopula has no bespoke Marshall–Olkin frailty
# sampler; instead we draw via the inverse Rosenblatt transform, which is driven
# by our closed-form `_cdf`/`DistortionFromCop`/`subsetdims` specialisations
# (nested/NestedConditioning.jl). A single VECTOR `_rand!` is enough — the
# Distributions.jl `rand(C, n)` matrix path loops over this vector form (mirrors
# ArchimedeanCopula/SubsetCopula). Dispatches only on NestedArchimedeanCopula, so
# the flat-collapse ArchimedeanCopula sampler is untouched. The per-coordinate
# inverse-CDF is an O(d) sequential bisection (≈ms/sample); keep sampled N modest.
function Distributions._rand!(rng::Distributions.AbstractRNG,
                              C::NestedArchimedeanCopula{d},
                              x::AbstractVector{T}) where {T<:Real, d}
    x .= inverse_rosenblatt(C, rand(rng, T, d))
    return x
end

# Copula-scale mixed partial of the nested CDF over the observed coordinates.
# Internal numerator kernel for the nested condition/subsetdims fast path
# (`DistortionFromCop(::NestedArchimedeanCopula, …)` in nested/NestedConditioning.jl);
# it works on copula-scale arguments `u ∈ (0,1)^d`. `censored[i] == true` means
# coordinate `i` enters only the argument-sum (not differentiated).
function _censored_copula_logpdf(C::NestedArchimedeanCopula{d}, u, censored, ::Type{T}) where {d, T}
    tree = _build_tree(C, u, collect(Bool, censored), T)
    return _nested_logpdf(tree)
end

Base.show(io::IO, C::NestedArchimedeanCopula{d}) where {d} =
    print(io, "NestedArchimedeanCopula{$d}($(C.G), $(length(C.children)) children)")
