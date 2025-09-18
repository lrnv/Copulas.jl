"""
    Copula{d} <: Distributions.ContinuousMultivariateDistribution

Abstract super–type for all `d`‑dimensional copula distributions in `Copulas.jl`.

A copula is a multivariate distribution on the unit hypercube `[0,1]^d` with
uniform(0,1) margins. Concrete subtypes (Archimedean, Elliptical, Extreme‑Value,
Archimax, miscellaneous constructions, …) implement the low–level API required
by the generic methods defined here:

Core interface methods expected (some have defaults):
* `cdf(C, u)` – cumulative distribution function on `[0,1]^d`.
* `pdf(C, u)` – density (when it exists) on `(0,1)^d`.
* `rand(rng, C)` – draw a single sample; vectorized `rand(rng, C, n)` should work via broadcasting.
* `Base.length(C)` – returns the dimension `d` (provided here).
* Dependence summaries: Kendall's τ (`τ(C)` / `StatsBase.corkendall(C)`) and
    Spearman's ρ (`ρ(C)` / `StatsBase.corspearman(C)`). Generic numeric
    integration / Monte‑Carlo fallbacks are provided and can be specialized.

Extended helper API supplied by:
* `measure(C, a, b)` – copula probability (rectangular volume) over the axis–aligned box
    `[a,b] ⊂ [0,1]^d` using inclusion–exclusion (exact via 2^d evaluations) with a Gray‑code loop.
* `subsetdims(C, dims)`  – marginal / reduced–dimension copula.
* `condition(C, dims, values)`  – conditional copula. 

Concrete subtype authors SHOULD at least implement an efficient `_cdf` and eventually implement `Distributions.pdf` or `Distributions._logpdf` and `Distributions._rand!`.

See also: [`ArchimedeanCopula`](@ref), [`EllipticalCopula`](@ref), [`ExtremeValueCopula`](@ref),
[`ArchimaxCopula`](@ref), [`measure`](@ref).

References
* [nelsen2007](@cite) Nelsen (2007), *An Introduction to Copulas*.
* [mcneil2009](@cite) McNeil & Nešlehová (2009), Multivariate Archimedean copulas, d‑monotone functions and ℓ₁‑norm symmetric distributions.
"""
abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.broadcastable(C::Copula) = Ref(C)
Base.length(::Copula{d}) where d = d

# Generic CDF (no docstring per style directive).
function Distributions.cdf(C::Copula{d},u::VT) where {d,VT<:AbstractVector}
    length(u) != d && throw(ArgumentError("Dimension mismatch between copula and input vector"))
    if any(iszero,u)
        return zero(u[1])
    elseif all(isone,u)
        return one(u[1])
    end
    return _cdf(C,u)
end
function Distributions.cdf(C::Copula{d},A::AbstractMatrix) where d
    size(A,1) != d && throw(ArgumentError("Dimension mismatch between copula and input vector"))
    return [Distributions.cdf(C,u) for u in eachcol(A)]
end

# Internal CDF fallback (undocumented by directive).
function _cdf(C::CT,u) where {CT<:Copula}
    f(x) = Distributions.pdf(C,x)
    z = zeros(eltype(u),length(C))
    return HCubature.hcubature(f,z,u,rtol=sqrt(eps()))[1]
end

"""
    ρ(C::Copula{d}) -> Real

Spearman's ρ of copula `C` computed by numeric integration of the CDF:

ρ = 12 ∫_{[0,1]^d} C(u) du  - 3.

For bivariate copulas this reduces to the classical definition.
Specialize for performance when a closed form (or fast Monte‑Carlo) is available.

References
* [nelsen2007](@cite) Nelsen (2007), *An Introduction to Copulas*.
"""
function ρ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    z = zeros(d)
    i = ones(d)
    r = HCubature.hcubature(F,z,i,rtol=sqrt(eps()))[1]
    return 12*r-3
end

"""
    τ(C::Copula) -> Real

Kendall's τ of copula `C` estimated by Monte‑Carlo using `Distributions.expectation`.

Definition:
τ = 4 E[ C(U) ] - 1,  where U ∼ C.

The default uses `nsamples = 10^4`. Override / specialize for analytic formulae
(e.g. Archimedean, Elliptical) or higher precision.

References
* [nelsen2007](@cite) Nelsen (2007), *An Introduction to Copulas*.
"""
function τ(C::Copula)
    F(x) = Distributions.cdf(C,x)
    r = Distributions.expectation(F,C; nsamples=10^4)
    return 4*r-1
end
"""
    StatsBase.corkendall(C::Copula{d}) -> Matrix{Float64}

Matrix of pairwise (bivariate) Kendall τ values between all component pairs of `C`.
Implemented by forming 2‑dimensional sub‑copulas via `SubsetCopula` and calling `τ`.

Complexity: O(d²) evaluations of `τ` (which may themselves be Monte‑Carlo if not
specialized). Override for structured copulas if a faster block formula exists.
"""
function StatsBase.corkendall(C::Copula{d}) where d
    # returns the matrix of bivariate kendall taus.
    K = ones(d,d)
    for i in 1:d
        for j in i+1:d
            K[i,j] = τ(SubsetCopula(C::Copula{d},(i,j)))
            K[j,i] = K[i,j]
        end
    end
    return K
end
"""
    StatsBase.corspearman(C::Copula{d}) -> Matrix{Float64}

Matrix of pairwise Spearman ρ values between all component pairs of `C`.
Implemented by building 2D marginals and calling `ρ`.

See also: [`ρ`](@ref), [`StatsBase.corkendall`](@ref).
"""
function StatsBase.corspearman(C::Copula{d}) where d
    # returns the matrix of bivariate spearman rhos.
    K = ones(d,d)
    for i in 1:d
        for j in i+1:d
            K[i,j] = ρ(SubsetCopula(C::Copula{d},(i,j)))
            K[j,i] = K[i,j]
        end
    end
    return K
end
"""
    measure(C::Copula{d}, a, b) -> Real

Probability mass of the axis‑aligned box `[a,b] ⊂ [0,1]^d` under copula `C`:

Pr(a₁ ≤ U₁ ≤ b₁, …, a_d ≤ U_d ≤ b_d) = Σ_{ε∈{0,1}^d} (-1)^{d-|ε|} C(w_ε),
where each corner `w_ε` uses `b_i` if `ε_i=1` else `a_i`.

Implementation details:
* Uses Gray‑code iteration to flip a single coordinate per corner, minimizing allocations.
* Returns `0` if any upper bound ≤ lower bound; returns `1` if the box is the whole hypercube.
* Clamps inputs to `[0,1]`.

See also: [`cdf`](@ref), [`subsetdims`](@ref).
References
* [cherubini2009](@cite) Cherubini & Romagnoli (2009), Computing the volume of n‑dimensional copulas.
"""
function measure(C::Copula{d}, us,vs) where {d}

    # Computes the value of the cdf at each corner of the hypercube [u,v]
    # To obtain the C-volume of the box.
    # This assumes u[i] < v[i] for all i
    # Based on Computing the {{Volume}} of {\emph{n}} -{{Dimensional Copulas}}, Cherubini & Romagnoli 2009

    # We use a gray code according to the proposal at https://discourse.julialang.org/t/looping-through-binary-numbers/90597/6

    T = promote_type(eltype(us), eltype(vs), Float64)
    u = ntuple(j -> clamp(us[j], 0, 1), d)
    v = ntuple(j -> clamp(vs[j], 0, 1), d)
    any(v .≤ u) && return T(0)
    all(iszero.(u)) && all(isone.(v)) && return T(1)

    eval_pt = collect(u)
    # Inclusion–exclusion: the sign for the corner at u is (-1)^d
    # (for d even it's +1, for d odd it's -1). The Gray-code loop below
    # then applies alternating signs matching (-1)^(d - |ε|) as bits flip.
    sign = isodd(d) ? -one(T) : one(T)
    r = sign * Distributions.cdf(C, eval_pt)
    graycode = 0    # use a gray code to flip one element at a time
    which = fill(false, d) # false/true to use u/v for each component (so false here)
    for s = 1:(1<<d)-1
        graycode′ = s ⊻ (s >> 1)
        graycomp = trailing_zeros(graycode ⊻ graycode′) + 1
        graycode = graycode′
        eval_pt[graycomp] = (which[graycomp] = !which[graycomp]) ? v[graycomp] : u[graycomp]
        sign *= -1
        r += sign * Distributions.cdf(C, eval_pt)
    end
    return max(r,0)
end
"""
    measure(C::Copula{2}, a, b)

Specialized 2D version of [`measure(::Copula{d}, a, b)`](@ref) using the
inclusion–exclusion formula directly:
`C(v1,v2) - C(v1,u2) - C(u1,v2) + C(u1,u2)` after clamping.
"""
function measure(C::Copula{2}, us, vs)
    T = promote_type(eltype(us), eltype(vs), Float64)
    u1 = clamp(T(us[1]), 0, 1)
    u2 = clamp(T(us[2]), 0, 1)
    v1 = clamp(T(vs[1]), 0, 1)
    v2 = clamp(T(vs[2]), 0, 1)
    (v1 <= u1 || v2 <= u2) && return zero(T)
    u1 == 0 && u2 == 0 && v1 == 1 && v2 == 1 && return one(T)
    c11 = Distributions.cdf(C, [v1, v2])
    c10 = Distributions.cdf(C, [v1, u2])
    c01 = Distributions.cdf(C, [u1, v2])
    c00 = Distributions.cdf(C, [u1, u2])
    r = c11 - c10 - c01 + c00
    return max(r, T(0))
end