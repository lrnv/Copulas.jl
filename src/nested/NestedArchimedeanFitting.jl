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
    fit(CopulaModel, C0::NestedArchimedeanCopula, U; method=:mle, quick_fit=false, kwargs...)
    fit(C0::NestedArchimedeanCopula, U; kwargs...) -> NestedArchimedeanCopula

Maximum-likelihood estimation of the generator parameters of a nested
Archimedean copula, holding the tree structure FIXED.

`C0` is a template instance whose tree shape (leaf layout, children blocks) and
per-node generator families are kept fixed; only the scalar θ of each generator
(root and every inner node) is re-optimised. `U` is a `d×n` matrix of
pseudo-observations (columns = observations).

The optimisation runs in unconstrained space through a *parametrisation* — a map
`α -> NestedArchimedeanCopula` that is decoupled from the generator objects:

  * **default**: each generator is reparametrised independently inside its own
    family domain (per-family `_unbound_params`/`_rebound_params`). The cross-node
    nesting condition is NOT enforced — the constructor leaves it to the caller.
  * **`reparam = (α -> C), init = α₀`**: supply your own parametrisation — to share
    parameters across nodes, change the per-generator parametrisation (e.g. fit on
    a Kendall-τ scale), or enforce a constraint such as nesting (parametrise each
    child's θ as a non-negative increment over its parent's). `reparam` must build
    the tree from `α` generically so ForwardDiff can differentiate it.

The two-argument form is a quick shim returning only the fitted copula.
"""
function Distributions.fit(::Type{CopulaModel}, C0::NestedArchimedeanCopula{d}, U;
        method=:mle, reparam=nothing, init=nothing,
        quick_fit=false, vcov=false, derived_measures=true, kwargs...) where {d}
    method === :mle || throw(ArgumentError("NestedArchimedeanCopula supports only method=:mle (got $method)."))
    size(U, 1) == d || throw(ArgumentError("Data dimension $(size(U,1)) ≠ copula dimension $d."))
    n = size(U, 2)

    # Resolve the parametrisation: an unconstrained α₀ and a map `recon: α -> copula`.
    α₀, recon = if reparam !== nothing
        init === nothing && throw(ArgumentError("`init` (the α₀ vector) is required alongside a custom `reparam`."))
        (collect(float.(init)), reparam)
    else
        (_nested_unbound(C0), Base.Fix1(_nested_rebound, C0))
    end

    loss(α) = -Distributions.loglikelihood(recon(α), U)

    t = @elapsed res = try
        Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff = ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss, α₀, Optim.NelderMead())
    end

    Chat = recon(Optim.minimizer(res))
    quick_fit && return (result = Chat,)

    ll = Distributions.loglikelihood(Chat, U)
    # NOTE: we deliberately do NOT put :θ̂ in the metadata, so the generic vcov
    # path (which reconstructs via the type-positional _unbound/_rebound and
    # `CT(d, θ...)` we do not have) is never reached.
    md = (; d, n, method = :mle, nparams = length(α₀),
          optimizer  = Optim.summary(res),
          converged  = Optim.converged(res),
          iterations = Optim.iterations(res),
          elapsed_sec = t, derived_measures, U = U)
    return CopulaModel(Chat, n, ll, :mle;
        vcov = nothing,
        converged = Optim.converged(res),
        iterations = Optim.iterations(res),
        elapsed_sec = t,
        method_details = md)
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

# Quick instance shortcut: returns only the fitted copula.
Distributions.fit(C0::NestedArchimedeanCopula, U; kwargs...) =
    Distributions.fit(CopulaModel, C0, U; quick_fit = true, kwargs...).result
