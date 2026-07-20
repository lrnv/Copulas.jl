# =============================================================================
# Density kernels for nested Archimedean copulas via Faà di Bruno's formula.
#
# A nested Archimedean copula is a tree of Archimedean generators: an outer
# generator `G` acts on a sum of inverse-generator terms, some of which are bare
# marginal coordinates and some of which are the CDFs of *inner* nested copulas.
# Its density is the mixed partial of that nested CDF over the differentiated
# (observed) coordinates.
#
# Differentiating the composition of generators is exactly Faà di Bruno's
# formula; the bookkeeping is carried by the partial Bell polynomials, here
# expressed through truncated Taylor series. Concretely, for each node we form
# the Taylor expansion of the inner-to-outer change of variables `ϕ⁻¹_outer ∘
# ϕ_inner`, raise it to the appropriate power (the leaf-count of the child),
# convolve the children's contributions (a Cauchy product of the per-child
# Taylor coefficient vectors), and finally contract the resulting coefficient
# vector against the derivatives `ϕ⁽ᵏ⁾` of the outer generator. The same
# recursion, restricted to differentiating the observed coordinates only, yields
# lower-tail partial-observation likelihoods; right-tail censoring is represented
# by applying the same recipe to a `SurvivalCopula`.
#
# Follows the nested-density and partial-observation likelihood algorithm of
# Yang & Li, "Archimedean Copula Inference via Taylor-Mode AD," arXiv:2605.23134
# (2026); see the `NestedArchimedeanCopula` docstring.
#
# Everything is written generically in the value type `T`. The default working
# type is `Float64`; passing `BigFloat` (or any high-precision real, e.g.
# `Double64`) coordinates flows that precision through the whole recursion, which
# is the recommended option for adversarial high-dimensional or deep-tail inputs
# where the alternating-sign Faà di Bruno sum can lose Float64 precision.
#
# These helpers build *only* on the public Copulas generator interface
# (`ϕ`, `ϕ⁻¹`, `ϕ⁽ᵏ⁾`, `ϕ⁻¹⁽¹⁾`); they do not introduce a parallel generator
# system. The Taylor route requires `ϕ` (and, for nested nodes, `ϕ⁻¹` of the
# outer generator) to accept a `TaylorSeries.Taylor1` argument; that holds for
# the closed-form generators shipped with the package.
# =============================================================================

# Internal tree representation built from a NestedArchimedeanCopula and a data
# point. A node carries its outer generator, its bare leaves (each a coordinate
# value and a censoring flag), and its child sub-trees.
struct _NestedLeaf{T}
    u::T
    censored::Bool
end
struct _NestedNode{TG<:Generator, T}
    G::TG
    leaves::Vector{_NestedLeaf{T}}
    children::Vector{_NestedNode}
end

"""
    composition_taylor(outer::Generator, inner::Generator, t₀, d) -> Vector

Overridable hook for the parent→child edge composition in a nested Archimedean
density. Returns the Taylor coefficients `[h⁽¹⁾(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!]` (the
constant term `h₀` dropped) of the inner-to-outer change of variables
`h = ϕ⁻¹_outer ∘ ϕ_inner`.

The default delegates to [`composition_taylor_direct`](@ref). Select a different
method, or supply your own, by adding a method to this function — most-specific
wins, no keyword or flag, mirroring the per-generator `ϕ⁽ᵏ⁾` override idiom:

  * **switch globally to the implicit solver** [`composition_taylor_implicit`](@ref)
    (paper App. A.4 — uses only scalar `ϕ⁽ᵏ⁾` and one scalar `ϕ⁻¹`, never a
    `Taylor1` through `ϕ⁻¹`; the method to use when a generator's `ϕ⁻¹` has no
    `Taylor1` method):

    ```julia
    Copulas.composition_taylor(o::Copulas.Generator, i::Copulas.Generator, t₀, d) =
        Copulas.composition_taylor_implicit(o, i, t₀, d)
    ```

  * **register a closed form for a generator pair** (fastest, most robust — see the
    Clayton/Clayton method in `Generator/ClaytonGenerator.jl`).

  * **roll your own** with the [`taylor`](@ref Copulas.taylor) primitive: jet your
    (possibly hand-simplified) link and drop the constant term — `taylor` returns
    `[f(t₀), f'(t₀)/1!, …]`, so take `[2:d+1]`:

    ```julia
    Copulas.composition_taylor(o::MyGen, i::MyGen, t₀, d) =
        Copulas.taylor(t -> Copulas.ϕ⁻¹(o, Copulas.ϕ(i, t)), t₀, d)[2:d+1]
    ```

The working type flows from `t₀`, so `BigFloat`/`Double64` precision is carried
through whichever method is selected.
"""
composition_taylor(outer::Generator, inner::Generator, t₀, d::Int) =
    composition_taylor_direct(outer, inner, t₀, d)

"""
    composition_taylor_direct(outer, inner, t₀, d)

Default edge composition (see [`composition_taylor`](@ref)): a single Taylor jet
over the explicit composition `ϕ⁻¹_outer ∘ ϕ_inner` at `t₀`, returning the
coefficients `[h⁽ᵏ⁾(t₀)/k! for k in 1:d]`. Requires both `ϕ` and `ϕ⁻¹` to accept a
`Taylor1` argument.
"""
function composition_taylor_direct(outer::Generator, inner::Generator, t₀::T, d::Int) where {T}
    coefs = taylor(x -> ϕ⁻¹(outer, ϕ(inner, x)), t₀, d)
    return T[coefs[k+1] for k in 1:d]
end

"""
    composition_taylor_implicit(outer, inner, t₀, d)

Edge composition by implicit differentiation (paper App. A.4; see
[`composition_taylor`](@ref)): `h` satisfies `ϕ_outer(h(t)) = ϕ_inner(t)`, solved
order by order by a triangular system using only the scalar derivatives `ϕ⁽ᵏ⁾` of
both generators and one scalar `ϕ⁻¹_outer` — it never puts a `Taylor1` through
`ϕ⁻¹`, so it is the method to use when a generator's `ϕ⁻¹` has no `Taylor1` method.
Returns the same `[h⁽ᵏ⁾(t₀)/k! for k in 1:d]` convention as
[`composition_taylor_direct`](@ref).
"""
function composition_taylor_implicit(outer::Generator, inner::Generator, t₀::T, d::Int) where {T}
    # h₀ = ϕ⁻¹_outer(ϕ_inner(t₀)); with h(t₀+ε)=h₀+Q(ε), Q=Σ qₘεᵐ, aₘ=ϕ⁽ᵐ⁾(outer,h₀)/m!,
    # bₘ=ϕ⁽ᵐ⁾(inner,t₀)/m!, matching εᵏ in Σ aₘ Qᵐ = Σ bₘ εᵐ gives q₁=b₁/a₁ and the
    # triangular qₖ=(bₖ − Σ_{m≥2} aₘ[εᵏ]Qᵐ)/a₁ below — no Taylor1 on ϕ/ϕ⁻¹.
    # Derivative-side coefficients: a[m] = ϕ⁽ᵐ⁾(outer,h₀)/m!, b[m] = ϕ⁽ᵐ⁾(inner,t₀)/m!
    # for m = 1..d (scalar k-th derivatives — NO Taylor1 on ϕ/ϕ⁻¹).
    b  = T[ϕ⁽ᵏ⁾(inner, m, t₀) / T(factorial(big(m))) for m in 1:d]
    h₀ = ϕ⁻¹(outer, ϕ(inner, t₀))                      # single scalar inverse
    a  = T[ϕ⁽ᵏ⁾(outer, m, h₀) / T(factorial(big(m))) for m in 1:d]

    q = zeros(T, d)
    q[1] = b[1] / a[1]
    d == 1 && return q

    # Amortized O(d³) column-update ladder (exact triangular recurrence).
    # Q[j+1] = q_j is the coefficient of εʲ in Q (Q has no constant term, Q[1]=0).
    # C[m+1, j+1] = [εʲ] Q^{m+2}, maintained as Q's coefficients fill in.
    npw = d - 1                                        # number of powers Q²..Q^d
    Q = zeros(T, d + 1); Q[2] = q[1]                   # Q = q₁ε initially
    C = zeros(T, npw, d + 1)
    for m in 0:npw-1
        (m + 2) <= d && (C[m+1, (m+2)+1] = q[1]^(m+2)) # [ε^{m+2}] (q₁ε)^{m+2}
    end
    for k in 2:d
        # correction = Σ_{m=0}^{npw-1} a[m+2] · [εᵏ] Q^{m+2}  (column k of C, final).
        corr = zero(T)
        for m in 0:npw-1
            (m + 2) <= d && (corr += a[m+2] * C[m+1, k+1])
        end
        q[k] = (b[k] - corr) / a[1]
        Q[k+1] = q[k]
        # Fill column j = k+1 of every power: C[m+1, j+1] = Σ_{i=1}^{k} q_i · prevₘ[j-i+1],
        # with prevₘ = Q^{m+1} (Q for m=0, else the lower power row). All reads touch
        # columns ≤ k, already finalised in prior outer steps, so rows update in parallel.
        j = k + 1
        if j <= d
            for m in 0:npw-1
                prev = m == 0 ? Q : view(C, m, :)      # prevₘ = Q^{m+1}
                s = zero(T)
                for i in 1:k
                    ji = j - i
                    (1 <= ji) && (ji + 1 <= d + 1) && (s += Q[i+1] * prev[ji+1])
                end
                C[m+1, j+1] = s
            end
        end
    end
    return q
end

# Partial-Bell-polynomial step: given the Taylor coefficients `p` of the link
# h (without its constant term) and the inner contribution `β` (the child's own
# Faà di Bruno coefficient vector), return the order-`d` coefficient vector of
# the composed contribution: it combines the derivatives of the composition with
# the inner block's coefficients, with the partial Bell polynomial B_{n,k}
# accumulated implicitly through the running polynomial powers `q`.
function _faa_di_bruno_coeffs(p::AbstractVector{T}, β::AbstractVector{T}, d::Int) where {T}
    n = d + 1
    p_pad = zeros(T, n)
    p_pad[2:min(n, length(p) + 1)] .= p[1:min(d, length(p))]
    facts = T[factorial(big(j)) for j in 0:d]
    invf  = T[1 / factorial(big(j)) for j in 0:d]
    β_scaled = T[β[j] * facts[j] for j in 1:n]
    α = zeros(T, n)
    q1 = copy(p_pad)
    q2 = zeros(T, n)
    for k in 1:d
        s = zero(T)
        for j in 1:n
            s += β_scaled[j] * q1[j]
        end
        α[k + 1] = s * invf[k + 1]
        if k < d
            fill!(q2, zero(T))
            for i in 1:n, j in 1:n
                idx = i + j - 1
                idx <= n && (q2[idx] += q1[i] * p_pad[j])
            end
            q1, q2 = q2, q1
        end
    end
    return α
end

# Cauchy product (polynomial convolution) of the per-child coefficient vectors,
# truncated to order `d`. Combining sibling sub-trees under one parent generator
# multiplies their Faà di Bruno generating polynomials.
function _cauchy_product(coeffs::Vector{Vector{T}}, d::Int) where {T}
    n = d + 1
    isempty(coeffs) && return T[one(T)]
    r1 = zeros(T, n)
    a1 = coeffs[1]
    r1[1:min(n, length(a1))] .= a1[1:min(n, length(a1))]
    r2 = zeros(T, n)
    for i in 2:length(coeffs)
        a = coeffs[i]
        fill!(r2, zero(T))
        for j in 1:n, k in 1:min(length(a), n - j + 1)
            r2[j + k - 1] += r1[j] * a[k]
        end
        r1, r2 = r2, r1
    end
    return r1
end

# Process one node: accumulate its argument t = Σ ϕ⁻¹(leaf/child-CDF), the node
# CDF C = ϕ(G, t), the number `d` of differentiated coordinates in the subtree,
# and its Faà di Bruno coefficient vector `β`. Censored leaves shift `t` but do
# not add a differentiation order. Returns (t, C, d, β).
function _process_node(node::_NestedNode{TG, T}) where {TG, T}
    G = node.G
    psum = zero(T)
    coeffs = Vector{T}[]
    dtot = 0
    for leaf in node.leaves
        psum += ϕ⁻¹(G, leaf.u)
        if leaf.censored
            push!(coeffs, T[one(T)])
        else
            push!(coeffs, T[zero(T), one(T)])
            dtot += 1
        end
    end
    for child in node.children
        (tc, Cc, dc, βc) = _process_node(child)
        psum += ϕ⁻¹(G, Cc)
        if dc == 0
            push!(coeffs, T[one(T)])
        else
            p = composition_taylor(G, child.G, tc, dc)
            β_pad = zeros(T, dc + 1)
            β_pad[1:min(dc + 1, length(βc))] .= βc[1:min(dc + 1, length(βc))]
            push!(coeffs, _faa_di_bruno_coeffs(p, β_pad, dc))
            dtot += dc
        end
    end
    t = psum
    C = ϕ(G, t)
    β = dtot == 0 ? T[one(T)] : _cauchy_product(coeffs, dtot)
    return (t, C, dtot, β)
end

# Log-Jacobian of the marginal change of variables: Σ log|ϕ⁻¹′(u)| over the
# differentiated (observed) leaves, recursively over the tree.
function _leaf_log_jacobian(node::_NestedNode{TG, T}) where {TG, T}
    s = zero(T)
    for leaf in node.leaves
        leaf.censored || (s += log(abs(ϕ⁻¹⁽¹⁾(node.G, leaf.u))))
    end
    for child in node.children
        s += _leaf_log_jacobian(child)
    end
    return s
end

# Contract the root coefficient vector against the outer generator derivatives:
# the density on the generator scale is Σₖ β[k] ϕ⁽ᵏ⁾(G, t). Done directly in the
# working type `T` (no log-space), so a high-precision `T` keeps the alternating
# sum exact.
function _assemble_density(β::Vector{T}, G::Generator, t::T, d::Int) where {T}
    d == 0 && return ϕ(G, t)
    s = zero(T)
    for k in 0:d
        ϕᵏ = k == 0 ? ϕ(G, t) : ϕ⁽ᵏ⁾(G, k, t)
        s += β[k + 1] * ϕᵏ
    end
    return s
end

# Full nested log-density of a node, including the marginal log-Jacobian. With
# no censored leaves this is the ordinary nested-Archimedean log-density; with
# censored leaves it is the mixed partial of the nested CDF over the observed
# coordinates only — the numerator kernel behind the condition/subsetdims
# fast path (see nested/NestedConditioning.jl).
function _nested_logpdf(node::_NestedNode{TG, T}) where {TG, T}
    (t, C, d, β) = _process_node(node)
    return log(abs(_assemble_density(β, node.G, t, d))) + _leaf_log_jacobian(node)
end

# Nested CDF of a node: C(u) = ϕ_G(Σ_leaves ϕ⁻¹(u) + Σ_children ϕ⁻¹(C_child)).
# Closed form, no Faà di Bruno / numerical integration needed.
function _nested_cdf(node::_NestedNode{TG, T}) where {TG, T}
    G = node.G
    s = zero(T)
    for leaf in node.leaves
        s += ϕ⁻¹(G, leaf.u)
    end
    for child in node.children
        s += ϕ⁻¹(G, _nested_cdf(child))
    end
    return ϕ(G, s)
end



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

# Partial-observation likelihood

Lower-tail partial observation is an *emergent* capability of the standard
[`condition`](@ref) + [`subsetdims`](@ref) framework — there is no bespoke
likelihood function. For an observed set ``O`` and a lower-tail set ``C``,
the likelihood factorises as the "gist recipe"

```julia
logpdf(subsetdims(X, O), x_O) + logcdf(condition(X, O, x_O), x_C)
```

(on the [`SklarDist`](@ref) `X = SklarDist(C, margins)`), which equals the
observed-marginal densities times the mixed partial of the nested CDF over the
observed coordinates. The `subsetdims`/`condition` specialisations for this type
route both factors through the Faà di Bruno tree walk; the denominator
``c_O`` cancels, reproducing the raw mixed partial.

Right-censored coordinates use the same recipe after flipping the censored
coordinates with [`SurvivalCopula`](@ref): on the copula scale, evaluate the
conditional lower tail of `SurvivalCopula(C, censored_dims)` at `1 .- u_C`.

# Example

```julia
using Copulas: ClaytonGenerator   # generator types are not exported

# outer Clayton(2) over two inner Clayton panels on dims 1:2 and 3:4
C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
        children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
logpdf(C, [0.3, 0.5, 0.4, 0.6])

# lower-tail likelihood with dim 2 partially observed (observed O = {1,3,4}, C = {2}):
S = SklarDist(C, ntuple(_ -> Exponential(1.0), 4))
x = [0.7, 0.3, 0.5, 0.9]
logpdf(subsetdims(S, (1, 3, 4)), x[[1, 3, 4]]) +
    logcdf(condition(S, (1, 3, 4), x[[1, 3, 4]]), x[2])

# right-censored dim 2 on the copula scale:
u = [cdf(S.m[i], x[i]) for i in 1:4]
Cs = SurvivalCopula(C, (2,))
logpdf(subsetdims(C, (1, 3, 4)), u[[1, 3, 4]]) +
    logcdf(condition(Cs, (1, 3, 4), u[[1, 3, 4]]), 1 - u[2])
```

The density and the partial-observation likelihood follow the
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
    function nextblock(k)
        s = 1
        while any((s + j) in used for j in 0:k-1)
            s += 1
        end
        return s
    end
    for c in pending
        k = _subdim(c); ds = Int[]; s = nextblock(k)
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
        T = eltype(u) <: AbstractFloat ? eltype(u) : Float64
        return T(-Inf)
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




    # =============================================================================
# Conditioning / subsetting fast paths for NestedArchimedeanCopula.
#
# These specialise upstream's standard conditioning/subsetting framework
# (`subsetdims`, `condition`, `DistortionFromCop`, `SubsetCopula`) so that
# per-variable censoring becomes an EMERGENT capability of the standard API,
# routed through our O(d²) Faà di Bruno tree walk rather than ForwardDiff.
#
# The "gist recipe" for the per-variable (right-)censored survival likelihood,
# with observed set O and censored/unobserved set C:
#
#     logpdf(subsetdims(X, O), x_O)  +  logcdf(condition(X, O, x_O), x_C)
#       = Σ_{i∈O} log f_i(x_i)  +  log [ ∂^{|O|} C / ∏_{i∈O} ∂u_i ] / c_O(u_O)
#       + log c_O(u_O)
#       = Σ_{i∈O} log f_i(x_i)  +  log ∂^{|O|} C / ∏_{i∈O} ∂u_i,
#
# i.e. the observed-marginal densities times the copula's mixed partial over the
# observed coordinates — the correct per-variable censored likelihood. The
# denominator c_O cancels, so the recipe reproduces the raw mixed partial.
#
# Two fast specialisations live here:
#   (1) SubsetCopula(::NestedArchimedeanCopula, dims) — prunes the generator
#       tree to the observed marginal, returning a re-indexed (1:p)
#       NestedArchimedeanCopula (or a flat ArchimedeanCopula when no genuine
#       sub-nesting survives). Reached automatically from `subsetdims`. Its
#       existing `_logpdf` then yields the observed-marginal density c_O.
#   (2) DistortionFromCop(::NestedArchimedeanCopula, js, ujs, i) — the closed-form
#       conditional marginal U_i | U_js = u_js, whose cdf is the mixed partial
#       over js (with i and every other coord entering only as a CDF argument)
#       divided by c_O. Handles GENERAL p, so it also accelerates each
#       per-coordinate distortion that the generic ConditionalCopula constructor
#       builds for the multi-unobserved case.
#
# MULTI-conditioned-dim conditional CDF (|unobserved| ≥ 2): now ALSO routed to the
# fast tree walk by the `_partial_cdf(C::NestedArchimedeanCopula, is, js, uᵢₛ, uⱼₛ)`
# specialisation at the bottom of this file (section (3)). `ConditionalCopula`
# stores its inner copula in the abstract FIELD `C::Copula{D}` (Conditioning.jl),
# but `_cdf(::ConditionalCopula)` calls `_partial_cdf(CC.C, …)`, which dispatches
# on the RUNTIME type of `CC.C` — concretely a `NestedArchimedeanCopula` — so our
# method is selected without `ConditionalCopula` needing to carry the inner type.
# The override assembles `u` (js→uⱼₛ, is→uᵢₛ, others→1) and returns
# `exp(_censored_copula_logpdf(C, u, cens, T))` with `cens[k] = !(k ∈ js)`. Thus
# BOTH `DistortionFromCop.cdf` (single-conditioned) and `ConditionalCopula._cdf`
# (multi-conditioned) compute the conditional CDF with our O(d²) Faà di Bruno
# walk — ZERO ForwardDiff for ANY number of censored dims.
#
# CAVEAT (forward-compat only): end-to-end BigFloat MULTI-censored *conditioning*
# via `condition()` is not yet enabled — upstream's `ConditionalCopula`/
# `DistortionFromCop` `uⱼₛ`/`den` fields are Float64-typed, so the standard API
# delivers Float64 to the override. Threading `T` future-proofs it (and matches
# `_assemble`'s own promotion), but BigFloat currently flows only via direct
# kernel calls or the single-censored `NestedDistortion.logcdf` path.
# =============================================================================

# ---- (1) SubsetCopula: prune the tree to the observed marginal --------------

# Count the OBSERVED leaves (membership in the observed set `O`) in a subtree.
_obscount(ch::Tuple, O) = count(in(O), ch[2])                  # flat panel (cc, ds)
function _obscount(c::NestedArchimedeanCopula, O)
    n = count(in(O), c.leafdims)
    for ch in c.children
        n += _obscount(ch, O)
    end
    return n
end

# The single observed GLOBAL dim of a subtree known to have exactly one. Used by
# the 1-observed-leaf collapse: such a subtree marginalises to the Uniform margin
# of that one coordinate, so it degenerates to a bare leaf under the nearest kept
# ancestor's generator (the inner generator vanishes).
function _the_one_observed_dim(ch::Tuple, O)
    for j in ch[2]
        j in O && return j
    end
    error("subtree claimed to have one observed leaf but none found")
end
function _the_one_observed_dim(c::NestedArchimedeanCopula, O)
    for j in c.leafdims
        j in O && return j
    end
    for ch in c.children
        _obscount(ch, O) == 1 && return _the_one_observed_dim(ch, O)
    end
    error("subtree claimed to have one observed leaf but none found")
end

# Prune one node to its observed-marginal STRUCTURE, returning a
# NestedArchimedeanCopula whose dims are still GLOBAL (a partial / non-contiguous
# subset of 1:d). Built directly via the inner constructor (NOT the validating
# keyword constructor, which would reject non-contiguous dims); the surviving
# tree is relabelled to 1:p in one pass afterwards via `_remap_dims`.
#
# Rules per child subtree S, with obscount = #observed leaves in S:
#   obscount(S) == 0 → drop S (it marginalises to all-ones; contributes nothing).
#   obscount(S) == 1 → collapse: the one observed leaf detaches and re-attaches to
#                      THIS node as a bare leaf under THIS node's generator (the
#                      inner generator vanishes — valid by single-coordinate
#                      marginalisation-invariance of Archimedean copulas).
#   obscount(S) >= 2 → keep S, recursing.
function _prune_node(c::NestedArchimedeanCopula, O)
    newleaf = Int[j for j in c.leafdims if j in O]
    newkids = Any[]
    for ch in c.children
        if ch isa Tuple                                  # flat panel (cc, ds)
            cc, ds = ch
            k = count(in(O), ds)
            k == 0 && continue
            if k == 1
                push!(newleaf, _the_one_observed_dim(ch, O))   # collapse to bare leaf
                continue
            end
            kept = Int[j for j in ds if j in O]
            push!(newkids, (ArchimedeanCopula(length(kept), cc.G), kept))  # keep, drop unobserved
        else                                             # nested child
            k = _obscount(ch, O)
            k == 0 && continue
            if k == 1
                push!(newleaf, _the_one_observed_dim(ch, O))   # cascade collapse
                continue
            end
            pruned = _prune_node(ch, O)
            if isempty(pruned.children)
                # The pruned child has only bare leaves under its own generator;
                # keep it as a flat panel (cc, ds) so its generator is preserved.
                push!(newkids, (ArchimedeanCopula(length(pruned.leafdims), pruned.G),
                                copy(pruned.leafdims)))
            else
                push!(newkids, pruned)
            end
        end
    end
    surv = sort!(vcat(newleaf, [j for ch in newkids for j in _kid_dims(ch)]))
    return NestedArchimedeanCopula{length(surv), typeof(c.G)}(c.G, newleaf, newkids, surv)
end

# GLOBAL dims covered by a pruned child entry.
_kid_dims(ch::Tuple) = ch[2]
_kid_dims(ch::NestedArchimedeanCopula) = ch.dims

function SubsetCopula(C::NestedArchimedeanCopula{d, TG}, dims::NTuple{p, Int}) where {d, TG, p}
    # `subsetdims` short-circuits p==1 (Uniform) and the identity `dims==1:d`, and
    # asserts `p <= d` otherwise, so here 2 <= p <= d and `dims` may be a (possibly
    # full, p==d) reordering/permutation of the kept coordinates.
    O = Set{Int}(dims)
    pruned = _prune_node(C, O)        # NestedArchimedeanCopula on GLOBAL dims
    # Genuinely-nested → flat collapse: every survivor lands directly under the
    # root (no surviving children), so the observed marginal is a flat copula
    # under the root generator (exchangeable, so the request order is immaterial).
    isempty(pruned.children) && return ArchimedeanCopula(p, C.G)
    # Relabel each REQUESTED global dim to its position in `dims`, preserving the
    # requested order (`dims` may be a permutation of the surviving dims). All
    # requested dims survive, so this Dict covers exactly the surviving dims.
    remap = Dict(dims[i] => i for i in eachindex(dims))
    return _remap_dims(pruned, remap)
end

# ---- (2) DistortionFromCop: closed-form conditional marginal U_i | U_js ------

"""
    NestedDistortion{TC,p} <: Distortion

Closed-form conditional marginal `U_i | U_js = u_js` of a
[`NestedArchimedeanCopula`](@ref). Its `cdf(D, u_i)` is the mixed partial of the
nested CDF over the conditioned set `js` (with `i` and every other coordinate
entering only as CDF arguments), divided by the observed-marginal density
`c_O = pdf(subsetdims(C, js), u_js)`. This routes the numerator through our
O(d²) Faà di Bruno tree walk rather than ForwardDiff. Handles general `p`, so it
is reused for each per-coordinate distortion the generic `ConditionalCopula`
constructor builds in the multi-unobserved case.
"""
struct NestedDistortion{TC, p} <: Distortion
    C::TC
    i::Int
    js::NTuple{p, Int}
    ujs::NTuple{p, Float64}
    logden::Float64
end

function DistortionFromCop(C::NestedArchimedeanCopula{D}, js::NTuple{p, Int},
                          ujs::NTuple{p, Float64}, i::Int) where {D, p}
    # den = c_O = pdf of the observed marginal. Identical to upstream's generic
    # DistortionFromCop.den, so num/den stays consistent. For p==1 subsetdims
    # returns Uniform() ⇒ den = 1; otherwise it is our pruned-tree multivariate
    # pdf. Do NOT assert p==D-1: condition() builds DistortionFromCop for EVERY
    # i∉js, so in the multi-unobserved case p < D-1.
    den = p == 1 ? Distributions.pdf(subsetdims(C, js), ujs[1]) :
                   Distributions.pdf(subsetdims(C, js), collect(ujs))
    return NestedDistortion{typeof(C), p}(C, i, js, ujs, log(float(den)))
end

function Distributions.logcdf(D::NestedDistortion, ui::Real)
    # Boundary guards keep the generic Distortion.quantile bisection well-posed
    # and logcdf monotone: P(U_i ≤ 0 | ·) = 0 ⇒ logcdf = -Inf; P(U_i ≤ 1 | ·) = 1
    # ⇒ logcdf = 0.
    ui <= 0 && return -Inf
    ui >= 1 && return 0.0
    d = length(D.C)
    T = float(promote_type(typeof(ui), Float64))
    u = ones(T, d)
    for k in 1:length(D.js)
        u[D.js[k]] = T(D.ujs[k])
    end
    u[D.i] = T(ui)
    # Observed/differentiated = js only; dim i AND every other unobserved coord
    # are censored (enter the argument-sum only, no differentiation).
    cens = trues(d)
    for j in D.js
        cens[j] = false
    end
    return _censored_copula_logpdf(D.C, u, cens, T) - D.logden
end

Distributions.cdf(D::NestedDistortion, ui::Real) = exp(Distributions.logcdf(D, ui))

# ---- (3) Multi-conditioned-dim conditional CDF: route Site B through our kernel
#
# Override the generic `_partial_cdf(C, is, js, uᵢₛ, uⱼₛ)` (Conditioning.jl:31),
# the order-|js| mixed partial of `cdf(C, ·)` over the observed dims `js`. The
# generic body takes it by NESTING |js| `ForwardDiff.derivative` calls — Dual-of-
# Dual type explosion at compile time, O(2^|js|) at run time — so it is infeasible
# in high dimension (many observed coordinates). Our Faà di Bruno walk is
# polynomial and handles any order. `_cdf(::ConditionalCopula)` (Conditioning.jl:170)
# calls `_partial_cdf(CC.C, …)`; `CC.C` sits in an abstract field but dispatches on
# its RUNTIME type (concretely nested), so this method is selected without
# `ConditionalCopula` carrying the inner type — routing the multi-conditioned-dim
# (|unobserved| ≥ 2) conditional CDF through our walk, no ForwardDiff.
#
# Body: assemble `u` (js→uⱼₛ, is→uᵢₛ, others→1), differentiate exactly `js`
# (cens[k] = !(k∈js)), return exp(kernel). A CDF's mixed partial over a coordinate
# subset is a non-negative sub-density, so exp(log|·|) == the value. `T` is
# threaded for a future BigFloat upper layer; the standard API stores Float64.
function _partial_cdf(C::NestedArchimedeanCopula{D}, is, js, uᵢₛ, uⱼₛ) where {D}
    T = float(promote_type(eltype(typeof(uᵢₛ)), eltype(typeof(uⱼₛ))))
    u = _assemble(D, is, js, uᵢₛ, uⱼₛ)        # js→uⱼₛ, is→uᵢₛ, others→1
    cens = trues(D)
    for j in js
        cens[j] = false                       # differentiate ONLY the conditioned dims
    end
    return exp(_censored_copula_logpdf(C, u, cens, T))
end


# =============================================================================
# Maximum-likelihood fitting for NestedArchimedeanCopula with a FIXED tree.
#
# The tree SHAPE (leaf layout, children blocks) and the generator FAMILY at every
# node are part of the model and are NOT inferable from data; only the scalar
# parameters θ of each generator (root and inner) are optimised. fit() therefore
# dispatches on a TEMPLATE INSTANCE `C0::NestedArchimedeanCopula` that carries the
# full tree — neither `fit(NestedArchimedeanCopula, U)` (a bare type) nor
# `fit(typeof(C0), U)` can rebuild the tree, since the type parameters
# `{d, root-TG}` encode only the dimension and the root generator family.
#
# The optimiser runs in UNCONSTRAINED reparameterised α-space, exactly like the
# generic `_fit(::Type{<:Copula}, U, ::Val{:mle})` driver in Fitting.jl: each
# generator's params are mapped to ℝ^p by the per-family `_unbound_params` and
# back by `_rebound_params`. This (a) keeps every individual generator inside its
# own valid family domain at all times, (b) sidesteps the missing `_θ_bounds` for
# the BB families, and (c) needs no box-constraint machinery.
#
# NESTING VALIDITY: the DEFAULT parametrisation does not enforce the cross-node
# "inner at least as dependent as outer" condition (the constructor leaves it to
# the caller) — an unconstrained α only keeps each generator valid in its own
# family, so a fitted optimum CAN have inner θ < outer θ. To constrain it, pass a
# custom `reparam`/`init` that encodes the constraint (see fit()).
#
# The tree is walked in a fixed PRE-ORDER (root generator, then each child block
# in `children` declaration order; a flat child `(ArchimedeanCopula, dims)` inline,
# a nested child by recursion). `_nested_unbound` and `_nested_rebound` MUST use
# the SAME order and the SAME local-arity helper, or a Frank node (which switches
# parameterisation at local-arity 2 vs ≥3) would corrupt its α-slice.
# =============================================================================

# Bare UnionAll generator type from an instance, e.g. ClaytonGenerator{Float64}
# -> ClaytonGenerator. Reconstruct via `_gentype(G)(values(nt)...)`: the Generator
# type-call (Generator.jl) splats positional args in field order, which equals the
# order of `Distributions.params`.
_gentype(G::Generator) = typeof(G).name.wrapper

# Local arity = number of ϕ⁻¹ terms this generator sums = #direct leaves +
# #direct children (for a node) or block size (for a flat child). This is the `d`
# whose per-family validity bound the generator's _unbound/_rebound depend on
# (Clayton's −1/(d−1); AMH/GumbelBarnett critical values; Frank's d==2 vs d≥3).
# Passing the GLOBAL tree d would over-restrict inner generators. Clamped to ≥2
# so Clayton's 1/(d−1) is finite for a single-child node.
_local_arity(C::NestedArchimedeanCopula) = max(length(C.leafdims) + length(C.children), 2)

# ---- FLATTEN: tree generators -> unconstrained ℝ^p vector -------------------
function _nested_unbound(C::NestedArchimedeanCopula)
    α = Float64[]
    _push_node!(α, C)
    return α
end
function _push_node!(α, C::NestedArchimedeanCopula)
    dloc = _local_arity(C)
    append!(α, _unbound_params(_gentype(C.G), dloc, Distributions.params(C.G)))
    for ch in C.children
        if ch isa Tuple                       # (flat ArchimedeanCopula, dims)
            cc, ds = ch
            append!(α, _unbound_params(_gentype(cc.G), max(length(ds), 2),
                                       Distributions.params(cc.G)))
        else                                  # nested child
            _push_node!(α, ch)
        end
    end
    return α
end

# Block length of a generator's α-slice (single-sourced through _unbound_params).
_blocklen(G::Generator, dloc) =
    length(_unbound_params(_gentype(G), dloc, Distributions.params(G)))

# ---- REBUILD: same tree skeleton + new α -> NestedArchimedeanCopula ----------
# Consume α left-to-right in the IDENTICAL pre-order; rebuild every generator with
# its new θ while preserving leafdims / children dims / tree shape exactly.
_nested_rebound(C::NestedArchimedeanCopula, α::AbstractVector) =
    _rebuild_node(C, α, Ref(1))
function _rebuild_node(C::NestedArchimedeanCopula, α, i::Ref{Int})
    dloc = _local_arity(C)
    k = _blocklen(C.G, dloc)
    newG = _gentype(C.G)(values(_rebound_params(_gentype(C.G), dloc, α[i[]:i[]+k-1]))...)
    i[] += k
    newkids = Any[]
    for ch in C.children
        if ch isa Tuple
            cc, ds = ch
            dl = max(length(ds), 2)
            kk = _blocklen(cc.G, dl)
            ng = _gentype(cc.G)(values(_rebound_params(_gentype(cc.G), dl, α[i[]:i[]+kk-1]))...)
            i[] += kk
            push!(newkids, (ArchimedeanCopula(length(ds), ng), ds))
        else
            push!(newkids, _rebuild_node(ch, α, i))
        end
    end
    return NestedArchimedeanCopula{length(C.dims), typeof(newG)}(
               newG, copy(C.leafdims), newkids, copy(C.dims))
end

# ---- Fitting-interface opt-outs ---------------------------------------------
# Advertise NO type-based fitting methods. The generic GenericTests "Fitting
# interface" testset and the package's type-positional fit machinery
# (`CT(d, θ...)`, `_example(CT, d)`) cannot reconstruct a tree copula, so we keep
# them OFF for the nested type — `can_be_fitted` becomes false and that whole
# block is skipped. The real, supported fit() is the instance API below. This
# also stops the false advertising of :itau/:irho/:ibeta (meaningless for a tree).
_available_fitting_methods(::Type{<:NestedArchimedeanCopula}, d) = Tuple{}()

# Bare-type _example throws: there is no canonical tree without a template
# (mirrors ArchimedeanCopula's bare _example).
_example(::Type{NestedArchimedeanCopula}, d) =
    throw(ArgumentError("Cannot fit a NestedArchimedeanCopula from the bare type: " *
        "the tree shape and generator families are not inferable from data. " *
        "Pass a template instance, e.g. `fit(CopulaModel, C0, U)` or `fit(C0, U)`."))

# ---- Parametrization layer (decoupled α -> tree map) ------------------------
# fit() optimises an unconstrained vector α through a reconstruction map
# `recon : α -> NestedArchimedeanCopula`. The DEFAULT (`_nested_rebound`)
# reparametrises each generator independently inside its own family domain. A
# CUSTOM `recon` (keywords `reparam`/`init`) decouples α from the generator
# objects entirely, so the caller can share parameters across nodes, change the
# per-generator parametrisation, or encode a constraint such as nesting (e.g. a
# child θ = parent θ + softplus(δ) increment; see the docs). `recon` only has to
# build the tree from α generically, so ForwardDiff differentiates straight through.

# ---- The MLE on a TEMPLATE INSTANCE (fixed structure) -----------------------
"""
    fit(CopulaModel, C0::NestedArchimedeanCopula, U)        # template tree
    fit(CopulaModel, reparam, init, U)                      # custom parametrisation

Maximum-likelihood estimation of the generator parameters of a nested Archimedean
copula. `U` is a `d×n` matrix of pseudo-observations (columns = observations). The
optimiser runs in an unconstrained space through a *parametrisation* — a map
`α -> NestedArchimedeanCopula` decoupled from the generator objects — supplied in
one of two ways:

  * **template** `C0`: a template instance whose tree shape (leaf layout, children
    blocks) and per-node generator families are kept fixed; only the scalar θ of
    each node is re-optimised, each inside its own family domain. The cross-node
    nesting condition is NOT enforced — the constructor leaves it to the caller.
  * **custom** `reparam`, `init`: your own map `reparam(α) -> copula` and its
    initial `α₀` (no template needed — the map fully defines the tree). Use it to
    share parameters across nodes, change the per-generator parametrisation (e.g.
    fit on a Kendall-τ scale), or enforce a constraint such as nesting (parametrise
    each child's θ as a non-negative increment over its parent's). `reparam` must
    build the tree from `α` generically so ForwardDiff can differentiate it.

`fit(C0, U)` is a quick shim returning only the fitted copula; for the custom form
use `fit(CopulaModel, reparam, init, U).result`.
"""
# Shared optimiser + model assembly for a parametrisation `recon: α -> copula`.
function _fit_nested(recon, α₀::AbstractVector, U, d::Int, n::Int; quick_fit, derived_measures)
    loss(α) = -Distributions.loglikelihood(recon(α), U)
    t = @elapsed res = try
        Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff = ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss, α₀, Optim.NelderMead())
    end
    Chat = recon(Optim.minimizer(res))
    quick_fit && return (result = Chat,)
    ll = Distributions.loglikelihood(Chat, U)
    # NOTE: we deliberately do NOT put :θ̂ in the metadata, so the generic vcov path
    # (type-positional reconstruction we do not have) is never reached.
    md = (; d, n, method = :mle, nparams = length(α₀),
          optimizer = Optim.summary(res), converged = Optim.converged(res),
          iterations = Optim.iterations(res), elapsed_sec = t, derived_measures, U = U)
    return CopulaModel(Chat, n, ll, :mle; vcov = nothing,
        converged = Optim.converged(res), iterations = Optim.iterations(res),
        elapsed_sec = t, method_details = md)
end

# Default: reparametrise a fixed TEMPLATE tree (its shape + families are kept fixed,
# only the scalar θ of every node is optimised).
function Distributions.fit(::Type{CopulaModel}, C0::NestedArchimedeanCopula{d}, U;
        method=:mle, quick_fit=false, vcov=false, derived_measures=true, kwargs...) where {d}
    method === :mle || throw(ArgumentError("NestedArchimedeanCopula supports only method=:mle (got $method)."))
    size(U, 1) == d || throw(ArgumentError("Data dimension $(size(U,1)) ≠ copula dimension $d."))
    return _fit_nested(Base.Fix1(_nested_rebound, C0), _nested_unbound(C0), U, d, size(U, 2);
                       quick_fit, derived_measures)
end

# Custom parametrisation: a map `reparam : α -> NestedArchimedeanCopula` and its
# initial α₀ — NO template, the map fully defines the tree (so it can share
# parameters, change the per-generator parametrisation, or encode a constraint).
function Distributions.fit(::Type{CopulaModel}, reparam, init::AbstractVector, U;
        method=:mle, quick_fit=false, vcov=false, derived_measures=true, kwargs...)
    method === :mle || throw(ArgumentError("NestedArchimedeanCopula supports only method=:mle (got $method)."))
    α₀ = collect(float.(init))
    d  = length(reparam(α₀))::Int                 # dimension from the parametrisation itself
    size(U, 1) == d || throw(ArgumentError("Data dimension $(size(U,1)) ≠ copula dimension $d."))
    return _fit_nested(reparam, α₀, U, d, size(U, 2); quick_fit, derived_measures)
end

# ---- coef / coefnames for a fitted nested copula ----------------------------
# We do not store :θ̂ in method_details (it would trigger the generic vcov path,
# which reconstructs a tree copula via the type-positional `CT(d, θ...)` we lack).
# Supply the parameters directly from the fitted tree instead: the natural θ of
# every generator (root, then each child block, pre-order) for display. The number
# of FREE parameters (which can be < #generators when a custom `reparam` shares
# parameters) is `length(α₀)`, recorded as `nparams`; `dof` uses it so AIC/BIC stay
# correct under any parametrisation.
function _nested_coef(C::NestedArchimedeanCopula, tag::String = "G")
    names = String[]; vals = Float64[]
    for (k, v) in pairs(Distributions.params(C.G))
        push!(names, "$(tag).$(k)"); push!(vals, float(v))
    end
    for (ci, ch) in enumerate(C.children)
        if ch isa Tuple
            cc, _ = ch
            for (k, v) in pairs(Distributions.params(cc.G))
                push!(names, "$(tag)[$(ci)].$(k)"); push!(vals, float(v))
            end
        else
            n2, v2 = _nested_coef(ch, "$(tag)[$(ci)]")
            append!(names, n2); append!(vals, v2)
        end
    end
    return names, vals
end
StatsBase.coef(M::CopulaModel{<:NestedArchimedeanCopula}) = _nested_coef(M.result)[2]
StatsBase.coefnames(M::CopulaModel{<:NestedArchimedeanCopula}) = _nested_coef(M.result)[1]
StatsBase.dof(M::CopulaModel{<:NestedArchimedeanCopula}) =
    hasproperty(M.method_details, :nparams) ? M.method_details.nparams : length(StatsBase.coef(M))

# Quick template shim: returns only the fitted copula. (No `fit(reparam, init, U)`
# shim — with an untyped `reparam` it would be type piracy on `Distributions.fit`;
# use `fit(CopulaModel, reparam, init, U).result` for the custom case.)
Distributions.fit(C0::NestedArchimedeanCopula, U; kwargs...) =
    Distributions.fit(CopulaModel, C0, U; quick_fit = true, kwargs...).result
