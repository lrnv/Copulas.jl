"""
    NoTail

Corresponds to the case where the pickads function is identically One, which means no particular tail behavior.
"""
struct NoTail <: Tail end
A(::NoTail, t::NTuple{d, <:Real}) where d = one(eltype(t))
ExtremeValueCopula(d, ::NoTail) = IndependentCopula(d)
A(::NoTail, t::Real) = 1.0
dA(::NoTail, ::Real) = 0.0
d²A(::NoTail, ::Real) = 0.0
