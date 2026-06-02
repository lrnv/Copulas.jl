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
            push!(newkids, (cc, Int[j for j in ds if j in O]))  # keep, drop unobserved
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
