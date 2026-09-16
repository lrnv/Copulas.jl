# Fitting must never expose a nested Archimedean tree that bypassed the
# constructor-level nesting certificates. The existing raw rebuild remains useful
# as an internal objective candidate, but every public/default reconstruction is
# validated before it can escape the optimizer.

_nested_rebound_unchecked(C::NestedArchimedeanCopula, α::AbstractVector) =
    _rebuild_node(C, α, Ref(1))

function _nested_tree_is_certified(C::NestedArchimedeanCopula)
    for entry in C.children
        child = _nested_child(entry)
        _nested_status(C.G, child.G, length(child)) === _NESTING_VALID || return false
        child isa NestedArchimedeanCopula && !_nested_tree_is_certified(child) && return false
    end
    return true
end

# Numeric parameter vectors are the only form used by fitting. Keep the broader
# fallback for internal structural uses, but make the fitting reconstruction go
# through the ordinary validating constructor recursively.
function _nested_rebound(C::NestedArchimedeanCopula, α::AbstractVector{<:Real})
    raw = _nested_rebound_unchecked(C, α)
    return _validated_nested_tree(raw)
end

function _validated_nested_tree(C::NestedArchimedeanCopula)
    children = Any[]
    for child in C.children
        if child isa Tuple
            push!(children, child)
        else
            push!(children, _validated_nested_tree(child))
        end
    end
    return NestedArchimedeanCopula{length(C)}(C.G, copy(C.leafdims), children)
end

@inline _nested_infinite_objective(α) =
    isempty(α) ? Inf : oftype(first(α), Inf)

function _nested_fit_loss(C0::NestedArchimedeanCopula, α::AbstractVector, U)
    candidate = _nested_rebound_unchecked(C0, α)
    _nested_tree_is_certified(candidate) || return _nested_infinite_objective(α)
    return -Distributions.loglikelihood(candidate, U)
end

# The template fitting path passes Base.Fix1(_nested_rebound, C0). Specialize
# that path only: custom user reparameterizations keep their existing semantics
# and their errors are never swallowed as inadmissible nesting proposals.
function _fit_nested(recon::Base.Fix1{typeof(_nested_rebound)},
                     α₀::AbstractVector, U)
    C0 = recon.x
    loss(α) = _nested_fit_loss(C0, α, U)
    res = try
        Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff = ADTypes.AutoForwardDiff())
    catch
        Optim.optimize(loss, α₀, Optim.NelderMead())
    end
    α = collect(Optim.minimizer(res))
    return recon(α), α
end
