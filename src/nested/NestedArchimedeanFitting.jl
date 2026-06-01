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
# NESTING VALIDITY is NOT enforced. The constructor itself does not check the
# cross-node "inner at least as dependent as outer" condition
# (NestedArchimedeanCopula.jl docstring); fit() mirrors that contract — it is the
# caller's responsibility. Unconstrained α only guarantees each generator is valid
# in its own family; a same-family fitted optimum CAN have inner θ < outer θ.
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

The optimisation runs in unconstrained reparameterised space (per-family
`_unbound_params`/`_rebound_params`), so each generator stays inside its own
valid family domain. The cross-node nesting-validity condition (for same-family
nestings, inner at least as dependent as outer) is NOT enforced — exactly as the
constructor leaves it to the caller. If you need a valid copula off the data,
check the fitted parameters yourself.

The two-argument form is a quick shim returning only the fitted copula.
"""
function Distributions.fit(::Type{CopulaModel}, C0::NestedArchimedeanCopula{d}, U;
        method=:mle, quick_fit=false, vcov=false, derived_measures=true, kwargs...) where {d}
    method === :mle || throw(ArgumentError("NestedArchimedeanCopula supports only method=:mle (got $method)."))
    size(U, 1) == d || throw(ArgumentError("Data dimension $(size(U,1)) ≠ copula dimension $d."))
    n = size(U, 2)

    α₀ = _nested_unbound(C0)
    cop(α) = _nested_rebound(C0, α)
    loss(α) = -Distributions.loglikelihood(cop(α), U)

    t = @elapsed res = try
        Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff = ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss, α₀, Optim.NelderMead())
    end

    Chat = cop(Optim.minimizer(res))
    quick_fit && return (result = Chat,)

    ll = Distributions.loglikelihood(Chat, U)
    # NOTE: we deliberately do NOT put :θ̂ in the metadata, so the generic vcov
    # path (which reconstructs via the type-positional _unbound/_rebound and
    # `CT(d, θ...)` we do not have) is never reached.
    md = (; d, n, method = :mle,
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
# every generator (root, then each child block, pre-order). `dof = length(coef)`
# then reports the correct free-parameter count, so AIC/BIC are correct.
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

# Quick instance shortcut: returns only the fitted copula.
Distributions.fit(C0::NestedArchimedeanCopula, U; kwargs...) =
    Distributions.fit(CopulaModel, C0, U; quick_fit = true, kwargs...).result
