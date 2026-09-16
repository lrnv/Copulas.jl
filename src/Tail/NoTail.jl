"""
    NoTail()

Internal extreme-value tail for independence. Its Pickands dependence function
is `A(t)=1` and its stable tail dependence function is `ℓ(x)=sum(x)`, so
`ExtremeValueCopula(d, NoTail())` reduces to the independent copula. This
parameter-free implementation is used as a limiting representation and is not
part of the public component API.
"""
struct NoTail <: Tail end

@inline limit_kind(::NoTail, ::Val) = Π_LIMIT

Distributions.params(::NoTail) = (;)

_unbound_params(::Type{NoTail}, d, ::NamedTuple) = Float64[]
_rebound_params(::Type{NoTail}, d, ::AbstractVector) = (;)

A(::NoTail, t::NTuple{d, <:Real}) where d = one(eltype(t))
A(::NoTail, t::Real) = 1.0
ℓ(::NoTail, x) = sum(x)
dA(::NoTail, ::Real) = 0.0
d²A(::NoTail, ::Real) = 0.0

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:NoTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    Random.rand!(rng, X)
    return X
end
