"""
    NoTail

Corresponds to the case where the pickads function is identically One, which means no particular tail behavior.
"""
struct NoTail <: Tail end
A(::NoTail, t::NTuple{d, <:Real}) where d = one(eltype(t))
ExtremeValueCopula(d, ::NoTail) = IndependentCopula(d)