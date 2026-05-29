# =============================================================================
# Per-variable (right-)censored survival likelihood.
#
# For partially-observed multivariate data — e.g. multivariate survival times
# where some coordinates are right-censored — the correct likelihood is the
# mixed partial of the joint CDF over the OBSERVED coordinates only, with the
# censored coordinates left as plain CDF arguments (integrated out above their
# observation time rather than differentiated).
#
# Sklar's theorem factorises that quantity into the observed marginal densities
# and the copula's mixed partial over the observed coordinates:
#
#   log f_survival(x, δ)
#       = Σ_{i : observed} log f_i(x_i)
#       + log ∂^{|O|} C(u) / ∏_{i∈O} ∂u_i   evaluated at u = (F_1(x_1), …, F_d(x_d)),
#
# where O = {i : δ_i = false} is the observed set and the censored coordinates
# enter C only as arguments. This is the user-facing survival likelihood, exposed
# as `logpdf(S::SklarDist, x; censored = δ)`. With `δ` omitted (all observed) it
# is exactly the ordinary joint density.
#
# Crucially this does NOT route through `Distributions.censored`: a censored
# marginal places an atom at the censoring time, which Sklar maps to the copula
# boundary u = 1 — off the open domain (0,1)^d where the copula density lives —
# so `logpdf(SklarDist(C, (…, censored(m), …)), x)` returns -Inf. The
# mixed-partial-over-observed-coordinates construction below is the correct
# replacement.
#
# The copula's contribution is supplied by `_censored_copula_logpdf(C, u, δ, T)`,
# the log of the mixed partial of the copula CDF over the observed coordinates.
# It is implemented here generically for flat `ArchimedeanCopula` (via ϕ⁽ᵏ⁾) and
# in NestedArchimedeanCopula.jl for nested trees (via the Faà di Bruno
# recursion). Any `Copula{d}` that adds a `_censored_copula_logpdf` method gains
# the SklarDist survival likelihood for free.
# =============================================================================

# ---- Flat Archimedean copula: mixed partial over the observed coordinates ----
# For an Archimedean copula C(u) = ϕ(Σ_i ϕ⁻¹(u_i)), the mixed partial over an
# observed set O of size k is
#     ∂^k C / ∏_{i∈O} ∂u_i = ϕ⁽ᵏ⁾(Σ_i ϕ⁻¹(u_i)) · ∏_{i∈O} ϕ⁻¹′(u_i),
# the censored coordinates contributing only through the inner sum. (k = 0, i.e.
# all coordinates censored, gives log C(u).)
function _censored_copula_logpdf(C::ArchimedeanCopula{d, TG}, u, censored, ::Type{T}) where {d, TG, T}
    s = sum(ϕ⁻¹(C.G, T(u[i])) for i in 1:d)
    k = count(!, censored)
    val = k == 0 ? ϕ(C.G, s) : ϕ⁽ᵏ⁾(C.G, k, s)  # ϕ⁽⁰⁾ = ϕ when fully censored
    logjac = zero(T)
    for i in 1:d
        censored[i] || (logjac += log(abs(ϕ⁻¹⁽¹⁾(C.G, T(u[i])))))
    end
    return log(abs(val)) + logjac
end

"""
    censored_logpdf(C::Copula, u, censored; T = Float64)

Copula-scale per-variable right-censored log-likelihood: the log of the mixed
partial of the copula CDF over the **observed** coordinates only, evaluated at
`u ∈ (0,1)^d`. Coordinate `i` is right-censored when `censored[i] == true`; such
coordinates enter the CDF as plain arguments but are not differentiated.

This is the lower-level building block behind the survival likelihood
`logpdf(SklarDist(C, margins), x; censored = δ)`. Most users want the latter,
which adds the observed marginal densities and works on the original data scale;
use `censored_logpdf` when you already hold copula-scale pseudo-observations.

With `censored` all-`false` this equals `logpdf(C, u)`; with all-`true` it equals
`log cdf(C, u)`. The keyword `T` sets the working precision (default `Float64`);
pass `T = BigFloat` for adversarial high-dimensional or deep-tail inputs.

Implemented for flat [`ArchimedeanCopula`](@ref) (via `ϕ⁽ᵏ⁾`) and
[`NestedArchimedeanCopula`](@ref) (via the Faà di Bruno recursion).
"""
function censored_logpdf(C::Copula{d}, u::AbstractVector,
                         censored::AbstractVector{Bool}; T::Type = Float64) where {d}
    length(u) == d || throw(ArgumentError("length(u) = $(length(u)) ≠ copula dimension $d"))
    length(censored) == d || throw(ArgumentError("length(censored) = $(length(censored)) ≠ copula dimension $d"))
    all(0 .< u .< 1) || throw(ArgumentError("censored_logpdf expects copula-scale u ∈ (0,1)^d"))
    return _censored_copula_logpdf(C, u, collect(Bool, censored), T)
end

"""
    logpdf(S::SklarDist, x; censored = falses(length(S)), T = Float64)

Log-likelihood of the joint model `S = SklarDist(C, margins)` at the data point
`x`, with optional per-variable right-censoring.

* With `censored` omitted (or all `false`) this is the ordinary joint
  log-density `Σ_i log f_i(x_i) + log c(F_1(x_1), …, F_d(x_d))`.
* With `censored[i] == true`, coordinate `i` is right-censored at `x_i`: only
  the survival information `T_i > x_i` is known. The returned value is then the
  partially-censored survival log-likelihood

  ```math
  \\sum_{i\\,:\\,\\text{observed}} \\log f_i(x_i)
  \\;+\\; \\log \\frac{\\partial^{|O|} C(\\mathbf u)}{\\prod_{i\\in O}\\partial u_i}
  \\Bigg|_{\\mathbf u = (F_1(x_1),\\dots,F_d(x_d))},
  ```

  where `O = {i : censored[i] == false}` and the censored coordinates enter the
  copula CDF only as arguments. With all coordinates censored it reduces to
  `log cdf(S, x)`.

This is the correct right-censored survival likelihood. It deliberately does
**not** use `Distributions.censored` margins: a censored margin places an atom at
the censoring time that Sklar maps to the copula boundary `u = 1`, off the open
domain `(0,1)^d` on which the copula density is defined, so
`logpdf(SklarDist(C, (…, censored(m), …)), x)` returns `-Inf`.

The keyword `T` sets the working precision of the copula partial (default
`Float64`, matching the uncensored path); pass `T = BigFloat` for adversarial
high-dimensional or deep-tail inputs.

The copula factor is computed by [`censored_logpdf`](@ref); it is available for
any `Copula{d}` that implements the mixed-partial-over-observed-coordinates
(flat [`ArchimedeanCopula`](@ref) and [`NestedArchimedeanCopula`](@ref) both do).
"""
function Distributions.logpdf(S::SklarDist, x::AbstractVector;
                              censored::AbstractVector{Bool} = falses(length(S)),
                              T::Type = Float64)
    d = length(S)
    length(x) == d || throw(ArgumentError("length(x) = $(length(x)) ≠ model dimension $d"))
    length(censored) == d || throw(ArgumentError("length(censored) = $(length(censored)) ≠ model dimension $d"))
    if !any(censored)
        return Distributions._logpdf(S, x)
    end
    # Observed marginal densities + copula mixed partial over observed coords.
    u = [clamp(Distributions.cdf(S.m[i], x[i]), 0, 1) for i in 1:d]
    margin_ll = zero(float(eltype(u)))
    for i in 1:d
        censored[i] || (margin_ll += Distributions.logpdf(S.m[i], x[i]))
    end
    return margin_ll + _censored_copula_logpdf(S.C, u, collect(Bool, censored), T)
end
