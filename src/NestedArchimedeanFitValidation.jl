# Fitting-specific validation for template-based NestedArchimedeanCopula fits.
#
# `_nested_rebound` deliberately preserves a tree's already-placed global
# dimension labels and therefore uses the raw structural constructor. Template
# fitting must nevertheless enforce the public nesting certificates at every
# proposal and again at the final minimizer.

function _nested_tree_certified(C::NestedArchimedeanCopula)
    for entry in C.children
        child = _nested_child(entry)
        _nested_status(C.G, child.G, length(child)) === _NESTING_VALID || return false
        entry isa NestedArchimedeanCopula && !_nested_tree_certified(entry) && return false
    end
    return true
end

function _validate_nested_tree(C::NestedArchimedeanCopula)
    _validate_nested_edges(C.G, C.children)
    for entry in C.children
        entry isa NestedArchimedeanCopula && _validate_nested_tree(entry)
    end
    return C
end

function _nested_fit_candidate(recon::Base.Fix1, α)
    recon.f === _nested_rebound || throw(ArgumentError(
        "_nested_fit_candidate is only defined for the template nesting reconstruction"))
    candidate = recon(α)
    return _nested_tree_certified(candidate) ? candidate : nothing
end

# Template fits currently pass `Base.Fix1(_nested_rebound, C0)`. Keep the
# generic `_fit_nested` method for custom parametrizations, but specialize this
# reconstruction so invalid/unsupported nesting proposals become inadmissible
# objective points rather than escaping as apparently valid copulas.
function _fit_nested(recon::Base.Fix1, α₀::AbstractVector, U)
    recon.f === _nested_rebound ||
        return invoke(_fit_nested, Tuple{Any,AbstractVector,Any}, recon, α₀, U)

    function loss(α)
        candidate = _nested_fit_candidate(recon, α)
        candidate === nothing && return oftype(first(α), Inf)
        return -Distributions.loglikelihood(candidate, U)
    end

    # Do not catch reconstruction errors here. Expected nesting failures are
    # represented by `nothing` above; any other failure is a genuine bug or
    # numerical error and must propagate instead of being masked by a fallback.
    res = Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff = ADTypes.AutoForwardDiff())
    α = collect(Optim.minimizer(res))

    # Reconstruct once more through the ordinary template map and run the same
    # validators as public construction. If an optimizer ever reports an
    # inadmissible minimizer, fail explicitly instead of returning it.
    fitted = recon(α)
    _validate_nested_tree(fitted)
    return fitted, α
end
