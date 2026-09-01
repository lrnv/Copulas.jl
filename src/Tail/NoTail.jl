"""
    NoTail

Corresponds to the case where the pickads function is identically One, which means no particular tail behavior.
"""
struct NoTail <: Tail end
Distributions.params(::NoTail) = (;)
A(::NoTail, t::NTuple{d, <:Real}) where d = one(eltype(t))
A(::NoTail, t::Real) = 1.0
ℓ(::NoTail, x) = sum(x)
dA(::NoTail, ::Real) = 0.0
d²A(::NoTail, ::Real) = 0.0
