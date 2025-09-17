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
    TT = promote_type(T, typeof(t))
    t <= 0 && return one(TT)
    t >= G.r[end] && return zero(TT)
    S = zero(TT)
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
    TT = promote_type(T, typeof(t))
    t >= G.r[end] && return zero(TT)
    S = zero(TT)
    @inbounds for j in lastindex(G.r):-1:firstindex(G.r)
        rⱼ, wⱼ = G.r[j], G.w[j]
        t ≥ rⱼ && break
        zpow = d==2 ? one(t) : (1 - t / rⱼ)^(d-2)
        S += wⱼ * zpow / rⱼ
    end
    return - (d-1) * S
end

# Higher derivatives ϕ^{(k)}(t) for 1 ≤ k ≤ d-1; 0 for k ≥ d
function ϕ⁽ᵏ⁾(G::EmpiricalGenerator{d, T}, ::Val{k}, t::Real) where {d, T, k}
    TT = promote_type(T, typeof(t))
    (k >= d || t >= G.r[end]) && return zero(TT)
    k == 0 && return ϕ(G, t)
    k == 1 && return ϕ⁽¹⁾(G, t)
    
    S = zero(TT)
    coeff = Base.factorial(d - 1) / Base.factorial(d - 1 - k)  # falling factorial (c)_k
    sgn = isodd(k) ? -1 : 1
    @inbounds for j in lastindex(G.r):-1:firstindex(G.r)
        rⱼ, wⱼ = G.r[j], G.w[j]
        t ≥ rⱼ && break
        zpow = (d == k+1) ? one(t) :  (1 - t / rⱼ)^(d - 1 - k)
        S += wⱼ * zpow / rⱼ^k
    end
    r = sgn * coeff * S 
    return r
end

# Monotone inverse ϕ^{-1}(x) with segment bracketing on knots r
function ϕ⁻¹(G::EmpiricalGenerator{d, T}, x::Real) where {d, T}
    TT = promote_type(T, typeof(x))
    x >= 1 && return zero(TT)
    x <= 0 && return TT(G.r[end])

    # Precompute f at knots t = r[k]
    # f(0) = 1 ≥ x; find smallest k s.t. f(r[k]) ≤ x
    N = length(G.r)
    k = 1
    while k <= N && ϕ(G, G.r[k]) > x
        k += 1
    end
    if k == 1
        a = zero(TT); b = G.r[1]
    elseif k > N
        # Should not happen since f(r_N)=0 ≤ x, but guard anyway
        return TT(G.r[end])
    else
        a = G.r[k-1]; b = G.r[k]
    end
    # Bracketed root find on [a,b]
    return TT(Roots.find_zero(t -> ϕ(G, t) - x, (a, b); bisection=true))
end

# Derivative of inverse: (ϕ^{-1})'(x) = 1 / ϕ'(t) at t = ϕ^{-1}(x)
ϕ⁻¹⁽¹⁾(G::EmpiricalGenerator{d, T}, x::Real) where {d, T} = inv(ϕ⁽¹⁾(G, ϕ⁻¹(G, x)))

function ϕ⁽ᵏ⁾⁻¹(G::EmpiricalGenerator{d, T}, ::Val{p}, y; start_at=nothing) where {d, T, p}
    TT = promote_type(T, typeof(y))

    # Guard invalid/degenerate derivative orders
    if p == 0
        return TT(ϕ⁻¹(G, y))
    elseif p < 1 || p >= d
        # ϕ^{(p)} is identically zero for p ≥ d (on [0, r_max)),
        # so the inverse is undefined except at y = 0. We return r[end]
        # as a safe endpoint in all these ambiguous cases.
        return TT(G.r[end])
    end

    # Precompute f at knots t = r[k]
    # f(0) = 1 ≥ y; find smallest k s.t. f(r[k]) ≤ y
    N = length(G.r)

    vp = Val{p}()
    sign = iseven(p) ? 1 : -1

    # Outside feasible range: map to endpoints
    sign*y <= 0 && return TT(G.r[end])
    sign*y >= sign*ϕ⁽ᵏ⁾(G, vp, 0) && return TT(0)

    # Find bracketing segment [a,b] over knot intervals
    k = 1
    while k <= N && sign*ϕ⁽ᵏ⁾(G, vp, G.r[k]) > sign*y
        k += 1
    end
    if k == 1
        a = zero(TT); b = G.r[1]
    elseif k > N
        # Should not happen since f(r_N)=0 ≤ y, but guard anyway
        return TT(G.r[end])
    else
        a = G.r[k-1]; b = G.r[k]
    end

    # Bracketed root find on [a,b]
    return TT(Roots.find_zero(t -> ϕ⁽ᵏ⁾(G, vp, t) - y, (a, b); bisection=true))
end


