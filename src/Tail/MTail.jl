"""
    MTail

Corresponds to the MCopula viewed as an etreme value copula.
"""
struct MTail <: Tail end
A(::MTail, t::NTuple{d, <:Real}) where d = max(t)
ExtremeValueCopula(d, ::MTail) = MCopula(d)