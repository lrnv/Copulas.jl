"""
    MTail

Corresponds to the MCopula viewed as an etreme value copula.
"""
struct NoTail <: Tail end
A(::NoTail, t::NTuple{d, <:Real}) = max(t)
ExtremeValueCopula(::MTail) = MCopula()