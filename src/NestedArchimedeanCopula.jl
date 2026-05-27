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

`logpdf(C, u; censored = δ)` computes the per-variable right-censored
log-likelihood: the mixed partial of the nested CDF over the *observed*
coordinates only, with right-censored coordinates (`δ[i] == true`) entering the
CDF as plain arguments rather than being differentiated. With `censored`
omitted (all observed) this is the ordinary nested density; with all coordinates
censored it reduces to `log C(u)`. This is the survival-analysis copula
likelihood for partially observed multivariate event times.

# Example

```julia
# outer Clayton(2) over two inner Clayton panels on dims 1:2 and 3:4
C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
        children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
logpdf(C, [0.3, 0.5, 0.4, 0.6])
logpdf(C, [0.3, 0.5, 0.4, 0.6]; censored = [false, true, false, true])
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
    T = eltype(u) <: AbstractFloat ? float(eltype(u)) : Float64
    tree = _build_tree(C, u, falses(d), T)
    return _nested_logpdf(tree)
end

"""
    logpdf(C::NestedArchimedeanCopula, u; censored = falses(length(C)))

Log-density of the nested Archimedean copula `C` at `u`, with optional
per-variable right-censoring.

* With `censored` omitted (or all `false`) this is the ordinary nested-copula
  log-density.
* With `censored[i] == true`, coordinate `i` is treated as right-censored: it
  enters the copula CDF as a plain argument but is **not** differentiated. The
  returned value is then the mixed partial of the nested CDF over the observed
  coordinates only — the per-variable censored (survival) copula likelihood for
  partially observed multivariate event times. With all coordinates censored it
  reduces to `log C(u)`.

The keyword `T` (default `BigFloat`) sets the working precision of the censored
recursion; censored survival likelihoods are typically evaluated on
high-dimensional panels where the extra precision is cheap insurance. The
uncensored `logpdf(C, u)` path uses the element type of `u` (defaulting to
`Float64`).

This censored likelihood differs from `logpdf(SklarDist(C, margins), x)` with
`Distributions.censored` margins: the latter plugs a censored marginal *density*
into a fully-differentiated joint density, and so does not compute the
mixed-partial-over-observed-dimensions quantity needed for survival data.
"""
function Distributions.logpdf(C::NestedArchimedeanCopula{d}, u::AbstractVector;
                              censored::AbstractVector{Bool} = falses(d),
                              T::Type = BigFloat) where {d}
    length(u) == d || throw(ArgumentError("length(u) = $(length(u)) ≠ copula dimension $d"))
    length(censored) == d || throw(ArgumentError("length(censored) = $(length(censored)) ≠ copula dimension $d"))
    if !any(censored)
        # No censoring: defer to the standard (element-typed) density path.
        return Distributions._logpdf(C, u)
    end
    tree = _build_tree(C, u, collect(Bool, censored), T)
    return _nested_logpdf(tree)
end

Base.show(io::IO, C::NestedArchimedeanCopula{d}) where {d} =
    print(io, "NestedArchimedeanCopula{$d}($(C.G), $(length(C.children)) children)")
