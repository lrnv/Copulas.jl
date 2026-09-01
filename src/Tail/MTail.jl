"""
    MTail

Corresponds to the MCopula viewed as an etreme value copula.
"""
struct MTail <: Tail end
A(::MTail, t::NTuple{d, <:Real}) where d = maximum(t)
A(::MTail, t::Real) = max(t, one(t) - t)
ℓ(::MTail, x) = maximum(x)
@inline limit_kind(::MTail, ::Val) = M_LIMIT
ExtremeValueCopula(d, ::MTail) = MCopula(d)
ExtremeValueCopula{d}(::MTail) where {d} = MCopula{d}()
