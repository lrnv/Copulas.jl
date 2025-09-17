"""
    EmpiricalGenerator

Fit a d-monotone Archimedean generator nonparametrically from a pseudo-sample
by inverting the empirical Kendall distribution to estimate a discrete radial law R̂,
and defining ϕ(t) = E[(1 - t/R̂)₊^(d-1)].

Constructor

    EmpiricalGenerator(u)

Inputs
- `u::AbstractMatrix`: d×n matrix, columns are observations on the copula or marginal scale (does not matter).
  or raw data to be converted with `pseudos(u)` when `pseudo_values=false`.

Fields
- Type parameter `d` — target Archimedean dimension for d-monotonicity
- `r::Vector{T}` — support (radii), sorted ascending with last element 1.0
- `w::Vector{T}` — associated weights summing to 1

Notes
- The scale of R is not identifiable; we normalize the largest recovered radius to 1.0.
- The resulting generator is d-monotone and suitable for dimensions up to `d`.

References
- [mcneil2009multivariate](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d‑monotone functions and ℓ1‑norm symmetric distributions.
- [williamson1956](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23.
- [genest2011a](@cite) Genest, C., & Nešlehová, J. (2011). On the empirical multilinear copula process. Journal reference for Kendall-based estimation links.
"""
struct EmpiricalGenerator{d, T} <: Generator
    r::Vector{T}
    w::Vector{T}
    function EmpiricalGenerator{d, T}(r::AbstractVector, w::AbstractVector) where {d, T}
        rT = T.(r)
        wT = T.(w)

        # checks: 
        length(rT) == length(wT) || throw(ArgumentError("length(r) != length(w)"))
        !isempty(rT) || throw(ArgumentError("no atoms given"))
        all(isfinite, rT) && all(>(zero(T)), rT) || throw(ArgumentError("atoms must be positive and finite"))
        all(isfinite, wT) && all(>(zero(T)), wT) || throw(ArgumentError("Weights must be positive and finite"))
        
        # sort ascending if necessary
        if !issorted(rT)
            p = sortperm(rT)
            rT = rT[p]
            wT = wT[p]
        end
        
        # enforce last atom exactly 1 and weights sum to 1: 
        rT ./= rT[end]
        wT ./= sum(wT)
        return new{d, T}(rT, wT)
    end
end
function EmpiricalGenerator(u::AbstractMatrix)
    T, d = eltype(u), size(u, 1) 
    
    # Get unique kendall atoms and weights
    W = _kendall_sample(u)
    kw = StatsBase.proportionmap(W)
    x = collect(keys(kw))
    N = length(x)

    N == 1 && return  EmpiricalGenerator{d, T}([T(1.0)], [T(1.0)]) # isnt this generator known somewhere ? 

    sort!(x; rev=true)
    w = [kw[xi] for xi in x]
    r = zero(x)
    r[end] = 1
    r[end-1] = 1 - clamp(x[N-1] / w[N], 0, 1)^(1/(d-1))

    for k in (N-2):-1:1
        gk = function(y)
            s = 0.0
            @inbounds for j in (k+1):N
                z = 1.0 - y / r[j]
                if z > 0.0
                    s += w[j] * z^(d-1)
                end
            end
            return s
        end
        eps = 1e-14
        a, b = 0.0, max(r[k+1] - eps, 0.0)
        ga, gb = gk(a), gk(b)
        if !(ga + 1e-12 >= x[k] >= gb - 1e-12)
            a, b = 0.0, r[k+1]
        end
        r[k] = Roots.find_zero(y -> gk(y) - x[k], (a, b); verbose=false)
        r[k] = clamp(r[k], 0.0, r[k+1] - eps)
    end

    # Final checks and normalization are enforced by inner constructor
    return EmpiricalGenerator{d, T}(r, w)
end
Distributions.params(G::EmpiricalGenerator{d, T}) where {d, T} = (d = d, radii = G.r, weights = G.w)
max_monotony(::EmpiricalGenerator{d, T}) where {d, T} = d

function ϕ(G::EmpiricalGenerator{d, T}, t::Real) where {d, T}
    t <= 0 && return one(float(t))
    t >= G.r[end] && return zero(float(t))
    S = zero(promote_type(T, typeof(float(t))))
    @inbounds for j in lastindex(G.r):-1:firstindex(G.r)
        rⱼ, wⱼ = G.r[j], G.w[j]
        t >= rⱼ && break
        S += wⱼ * (1 - t / rⱼ)^(d - 1)
    end
    return S
end

# Return the fitted radial distribution (Williamson pre-image) when the request matches D
williamson_dist(G::EmpiricalGenerator{d, T}, ::Val{d}) where {d, T} = Distributions.DiscreteNonParametric(G.r, G.w)

# ===============================
#  Specialized derivatives & inverse
# ===============================

# First derivative ϕ'(t)
function ϕ⁽¹⁾(G::EmpiricalGenerator{d, T}, t::Real) where {d, T}
    if t >= G.r[end]
        return zero(float(t))
    end
    S = zero(promote_type(T, typeof(float(t))))
    tv = ForwardDiff.value(t)
    c = d - 1
    @inbounds for j in lastindex(G.r):-1:firstindex(G.r)
        rj = G.r[j]
        if tv < rj
            z = one(S) - t / rj
            if c == 1
                # power 0 => z^(0) = 1
                S += -G.w[j] / rj
            else
                S += -(c) * (G.w[j] / rj) * z^(c - 1)
            end
        else
            break
        end
    end
    return S
end

# Higher derivatives ϕ^{(k)}(t) for 1 ≤ k ≤ d-1; 0 for k ≥ d
function ϕ⁽ᵏ⁾(G::EmpiricalGenerator{d, T}, ::Val{k}, t::Real) where {d, T, k}
    if k >= d || t >= G.r[end]
        return zero(float(t))
    elseif k == 0
        return ϕ(G, t)
    end
    S = zero(promote_type(T, typeof(float(t))))
    tv = ForwardDiff.value(t)
    c = d - 1
    pow = c - k
    coeff = one(S) * (Base.factorial(c) / Base.factorial(c - k))  # falling factorial (c)_k
    sgn = (isodd(k) ? -one(S) : one(S))
    @inbounds for j in lastindex(G.r):-1:firstindex(G.r)
        rj = G.r[j]
        if tv < rj
            z = one(S) - t / rj
            term = G.w[j] * (rj^(-k))
            if pow == 0
                S += sgn * coeff * term
            else
                S += sgn * coeff * term * z^(pow)
            end
        else
            break
        end
    end
    return S
end

# Monotone inverse ϕ^{-1}(x) with segment bracketing on knots r
function ϕ⁻¹(G::EmpiricalGenerator{d, T}, x::Real) where {d, T}
    xx = clamp(float(x), 0.0, 1.0)
    xx >= 1 && return zero(xx)
    xx <= 0 && return G.r[end]
    # Precompute f at knots t = r[k]
    # f(0) = 1 ≥ xx; find smallest k s.t. f(r[k]) ≤ xx
    N = length(G.r)
    f_at = @inline k -> begin
        s = 0.0
        rk = G.r[k]
        @inbounds for j in (k+1):N
            s += float(G.w[j]) * (max(1 - rk / G.r[j], 0.0))^(d - 1)
        end
        s
    end
    k = 1
    while k <= N && f_at(k) > xx
        k += 1
    end
    if k == 1
        a = zero(xx); b = G.r[1]
    elseif k > N
        # Should not happen since f(r_N)=0 ≤ xx, but guard anyway
        return G.r[end]
    else
        a = G.r[k-1]; b = G.r[k]
    end
    # Bracketed root find on [a,b]
    return Roots.find_zero(t -> ϕ(G, t) - xx, (a, b); bisection=true)
end

# Derivative of inverse: (ϕ^{-1})'(x) = 1 / ϕ'(t) at t = ϕ^{-1}(x)
ϕ⁻¹⁽¹⁾(G::EmpiricalGenerator{d, T}, x::Real) where {d, T} = inv(ϕ⁽¹⁾(G, ϕ⁻¹(G, x)))

function ϕ⁽ᵏ⁾⁻¹(G::EmpiricalGenerator{d, T}, ::Val{k}, y; start_at=nothing) where {d, T, k}
    # Monotone inverse for higher derivatives of the empirical generator.
    # For 1 ≤ k ≤ d-1, ϕ^{(k)} is piecewise-polynomial and monotone on [0, r_max],
    # with ϕ^{(k)}(0) ≠ 0 and ϕ^{(k)}(r_max) = 0. This provides a robust bracket for
    # root-finding. For k ≥ d, ϕ^{(k)} ≡ 0 and inversion is undefined.
    k >= d && throw(ArgumentError("ϕ^{($k)} is identically zero for k ≥ d; cannot invert."))
    yy = float(y)
    a = zero(yy)
    b = G.r[end]
    fa = ϕ⁽ᵏ⁾(G, Val{k}(), a)
    fb = ϕ⁽ᵏ⁾(G, Val{k}(), b)  # should be 0
    # Ensure y is within [min(fa,fb), max(fa,fb)]
    lo, hi = min(fa, fb), max(fa, fb)
    (yy < lo - sqrt(eps(float(yy)))) && (yy = lo)
    (yy > hi + sqrt(eps(float(yy)))) && (yy = hi)
    return Roots.find_zero(t -> ϕ⁽ᵏ⁾(G, Val{k}(), t) - yy, (a, b); bisection=true)
end


