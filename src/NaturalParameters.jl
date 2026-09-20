###############################################################################
##### Natural model parameter representations
###############################################################################

# Paramorph names are the canonical logical names for ordinary parametric
# components. Keep array-valued natural parameters independent of internal
# storage so `params` behaves like the generic Copula implementation.
@inline _copy_parameter_value(value) =
    value isa AbstractArray ? copy(value) : value

function _named_parameter_values(component, p)
    return map(Paramorph.names(p)) do name
        _copy_parameter_value(getproperty(component, name))
    end
end

# In-package univariate generator families use Paramorph as their canonical
# natural parameter description. Structural generators and downstream custom
# generators retain the explicit/fallback representations declared in
# ArchimedeanCopula.jl.
function Distributions.params(
    C::ArchimedeanCopula{d,G},
) where {d,G<:AbstractUnivariateGenerator}
    return _named_parameter_values(C.G, Paramorph.param_space(G, d))
end

function Distributions.params(
    C::ArchimedeanCopula{d,G},
) where {d,G<:AbstractUnivariateFrailtyGenerator}
    return _named_parameter_values(C.G, Paramorph.param_space(G, d))
end

# Liouville exposes the generator's natural parameters followed by the positive
# Dirichlet vector described by its Paramorph space. A downstream generator that
# does not implement Paramorph keeps the pre-existing structural representation.
function Distributions.params(
    C::LiouvilleCopula{d,TG,Tα},
) where {d,TG<:Generator,Tα<:Real}
    applicable(Paramorph.param_space, TG, 2) || return (C.G, C.α)
    gvals = _named_parameter_values(C.G, Paramorph.param_space(TG, 2))
    return (gvals..., collect(C.α))
end

# Archimax natural parameters are the concatenation of the two component model
# representations. Going through those public component `params` methods is
# important for tails such as Tawn/AsymGalambos whose logical Paramorph names do
# not correspond one-for-one to stored fields.
function Distributions.params(
    C::ArchimaxCopula{d,TG,TT},
) where {d,TG<:Generator,TT<:Tail}
    gvals = Distributions.params(ArchimedeanCopula{d}(C.gen))
    tvals = Distributions.params(ExtremeValueCopula{d}(C.tail))
    return (gvals..., tvals...)
end

# Liebscher is structurally composite: its natural public representation is the
# constructor-order pair `(copulas, weights)`. Public constructors always store
# an AbstractMatrix weight container, making this strictly more specific than
# the representation inherited from the pre-Paramorph Liebscher implementation.
# The runtime Paramorph chart used by template fitting is deliberately separate
# and may omit fixed components or structural zero weights.
function Distributions.params(
    C::LiebscherCopula{d,CT,WT},
) where {d,CT,WT<:AbstractMatrix}
    return (C.copulas, copy(C.weights))
end

# Empirical plug-in state is fitted state, not a statistical parameter. These
# objects have the zero-dimensional Paramorph space `()` and expose the same
# zero-dimensional natural parameter representation.
Distributions.params(::EmpiricalCopula{d,MT}) where {d,MT<:AbstractMatrix} = ()
Distributions.params(::BetaCopula{d,MT}) where {d,MT<:AbstractMatrix} = ()
Distributions.params(::CheckerboardCopula{d,T}) where {d,T<:Real} = ()
