"""
    MTail()

Internal extreme-value tail for complete positive dependence. Its Pickands
dependence function is `A(t)=max(t,1-t)` in dimension two and its stable tail
dependence function is `ℓ(x)=maximum(x)`, yielding the comonotonic `MCopula`.
This parameter-free limiting representation is not part of the public
component API.
"""
struct MTail <: Tail end
Distributions.params(::MTail) = (;)
A(::MTail, t::NTuple{d, <:Real}) where d = maximum(t)
A(::MTail, t::Real) = max(t, one(t) - t)
ℓ(::MTail, x) = maximum(x)
@inline limit_kind(::MTail, ::Val) = M_LIMIT
