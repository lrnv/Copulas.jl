"""
    LogTail{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    LogCopula(θ)
    ExtremeValueCopula(LogTail(θ))

The (bivariate) Mixed extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\theta t^2 - \\theta t + 1, \\quad t \\in [0,1].
```
Special cases:

* θ = 0 ⇒ IndependentCopula

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct LogTail{T} <: Tail2
    θ::T
    function LogTail(θ)
        (0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in [0,1]"))
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const LogCopula{T} = ExtremeValueCopula{2, LogTail{T}}
LogCopula(θ) = ExtremeValueCopula(2, LogTail(θ))
Distributions.params(C::LogTail) = (C.E.θ,)

function A(E::LogTail, t::Real)
    tt = _safett(t)
    θ  = E.θ
    return θ*tt^2 - θ*tt + 1
end