"""
    CheckerboardCopula{d, MT}

Fields:
- `n::Int` — number of partitions per axis.
- `h::MT`  — hypercubic array of cell masses (∑h = 1; each slice along an axis sums to 1/n).
- `P::MT`  — inclusive prefix sums of `h` (for fast CDF evaluation).

Constructor:

    CheckerboardCopula(X; n=nothing, pseudo_values=true, smoothing=1e-16, maxiter=5000, atol=1e-12)

The empirical checkerboard copula in dimension ``d`` is defined as the multilinear extension of the empirical copula:

```math
C_n^{\\mathrm{cb}}(u) = \\sum_{k} w_k(u) \\, C_n(v_k),
```
where ``v_k`` are the grid corners and ``w_k(u)`` the interpolation weights.

Notes:

- If `n` is `nothing`, is used `n = clamp(round(Int, m^(1/p)), 2, 256)`.
- This is always a valid copula for any finite sample size `N`.
- Supports `cdf`, `logpdf` at observed points and random sampling.

References
* Neslehova (2007). *On rank correlation measures for non-continuous random variables*.
* Segers, Sibuya & Tsukahara (2017). *The empirical beta copula*. J. Multivariate Analysis, 155, 35-51.
* Fredricks & Hofert (2025). *On the checkerboard copula and maximum entropy*.
"""
struct CheckerboardCopula{d,MT<:AbstractArray{<:Real,d}} <: Copula{d}
    n::Int   # partitions by axis
    h::MT    # masses per cell (∑h=1; each slice along an axis = 1/n)
    P::MT    # inclusive prefixes of h (for fast CDF)
end

function CheckerboardCopula(X::AbstractMatrix{<:Real};
                            n::Union{Int,Nothing}=nothing,
                            pseudo_values::Bool=true,
                            smoothing::Real=1e-16,
                            maxiter::Int=5_000,
                            atol::Real=1e-12)
    U = _as_pxn(size(X,1), X)   # p×m
    p, m = size(U)
    n === nothing && (n = clamp(round(Int, m^(1/p)), 2, 256))

    Uu = if pseudo_values
        minU = minimum(U); maxU = maximum(U)
        if !(minU ≥ -sqrt(eps(Float64)) && maxU ≤ 1 + sqrt(eps(Float64)))
            throw(ArgumentError("Con pseudo_values=true, X debe estar ya en (0,1). Use pseudo_values=false o convierta con `pseudos(X)`."))
        end
        clamp.(Float64.(U), eps(Float64), 1 - eps(Float64))
    else
        pseudos(U)
    end

    H = zeros(Float64, ntuple(_->n, p))
    @inbounds for t in 1:m
        idx = ntuple(j -> _cell_and_tau(Uu[j, t], n)[1], p)  # 1..n por eje
        H[CartesianIndex(idx)] += 1.0
    end
    H ./= m

    _sinkhorn!(H; maxiter=maxiter, atol=atol, smoothing=smoothing)

    P = _prefix_sum(H)

    return CheckerboardCopula{p, typeof(H)}(n, H, P)
end

function Distributions.logpdf(C::CheckerboardCopula{d}, u::AbstractVector{<:Real}) where {d}
    @assert length(u) == d
    n = C.n
    idx = ntuple(j -> begin
        i, _ = _cell_and_tau(u[j], n)
        i
    end, d)
    mass = @inbounds C.h[CartesianIndex(idx)]
    dens = (mass <= 0) ? 0.0 : (mass * n^d)
    return (dens > 0) ? log(dens) : -Inf
end

function Distributions.cdf(C::CheckerboardCopula{d}, u::AbstractVector{<:Real}) where {d}
    @assert length(u) == d
    n = C.n
    idx = ntuple(j -> _cell_and_tau(u[j], n), d)  # (i_j, τ_j)
    i   = ntuple(j -> idx[j][1], d)
    τ   = ntuple(j -> idx[j][2], d)
    k   = ntuple(j -> i[j]-1, d)

    s0 = any(kj == 0 for kj in k) ? zero(eltype(C.h)) :
         _sum_box(C.P, ntuple(_->1,d), k)

    s = s0
    for bt in _bit_tuples(Val(d))
        # skip emptu
        if all(b==0 for b in bt); continue; end
        w = one(eltype(C.h))
        @inbounds for j in 1:d
            (bt[j]==1) && (w *= τ[j])
        end
        (w == 0) && continue
        lows  = ntuple(j -> (bt[j]==1 ? k[j]+1 : 1), d)
        highs = ntuple(j -> (bt[j]==1 ? k[j]+1 : k[j]), d)
        s += w * _sum_box(C.P, lows, highs)
    end
    return float(s)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CheckerboardCopula{d}, u::AbstractVector{T}) where {d,T<:Real}
    @assert length(u) == d
    n = C.n
    p = reshape(C.h, :)
    cat = Distributions.Categorical(p ./ sum(p))
    lin = rand(rng, cat)
    I = CartesianIndices(size(C.h))[lin]
    @inbounds for j in 1:d
        a,b = _cell_bounds(I[j], n)
        u[j] = a + rand(rng)*(b - a)
    end
    return u
end


# =========================
#   Internal utils
# =========================

@inline function _cell_and_tau(u::Real, n::Int)
    (u < 0 || u > 1) && throw(ArgumentError("u ∈ [0,1] requerido"))
    if u == 1
        return (n, 1.0)
    else
        t = n*u
        k = Int(floor(t))  # 0..n-1
        return (k+1, t - k)
    end
end

@inline _cell_bounds(i::Int, n::Int) = ((i-1)/n, i/n)

# inclusive prefix
function _prefix_sum(h::AbstractArray{T,N}) where {T,N}
    P = copy(h)
    for ax in 1:N
        P = cumsum(P; dims=ax)
    end
    return P
end

@generated function _bit_tuples(::Val{d}) where {d}
    W = NTuple{d,Int}[]
    for m in 0:(2^d-1)
        push!(W, ntuple(j -> (m >> (j-1)) & 1, d))
    end
    return :(($(W...),))
end

@inline function _getP(P::AbstractArray{T,N}, idxs::NTuple{N,Int}) where {T,N}
    @inbounds for j in 1:N
        if idxs[j] == 0
            return zero(T)
        end
    end
    @inbounds return P[CartesianIndex(idxs)]
end

function _sum_box(P::AbstractArray{T,N}, l::NTuple{N,Int}, h::NTuple{N,Int}) where {T,N}
    @inbounds for j in 1:N
        if h[j] < l[j]; return zero(T); end
    end
    total = zero(T)
    for bt in _bit_tuples(Val(N))
        idxs = ntuple(j -> (bt[j] == 1 ? h[j] : l[j]-1), N)
        sgn  = (-1)^(sum(bt[j]==0 for j in 1:N))
        total += sgn * _getP(P, idxs)
    end
    total
end

@inline _sliceview(A, ax, t) = @view A[(ntuple(j -> (j==ax ? t : Colon()), ndims(A)))...]

function _is_multistochastic(h::AbstractArray{T,d}; atol=1e-12) where {T,d}
    n = size(h,1)
    ndims(h) == d || throw(ArgumentError("ndims(h)=$(ndims(h)) ≠ d=$d"))
    all(s -> s == n, size(h)) || throw(ArgumentError("h debe ser n×…×n"))

    F = float(T); target = one(F)/n
    @inbounds for x in h
        (!isfinite(x) || x < -sqrt(eps(F))) && return false
    end
    @inbounds for ax in 1:d, t in 1:n
        S = sum(_sliceview(h, ax, t))
        if !isfinite(S) || abs(S - target) > atol
            return false
        end
    end
    S = sum(h)
    isfinite(S) && abs(S - one(F)) ≤ atol
end

function _sinkhorn!(h::AbstractArray{T,d};
                    maxiter::Int=5_000, atol::Real=1e-12, smoothing::Real=1e-16) where {T,d}
    n = size(h,1)
    ndims(h) == d || throw(ArgumentError("ndims(h)=$(ndims(h)) ≠ d=$d"))
    all(s -> s == n, size(h)) || throw(ArgumentError("h debe ser n×…×n"))

    F = float(T)
    ε   = F(smoothing)
    invN  = one(F)/n
    invNd = one(F)/(n^d)

    @. h = (one(F) - ε)*h + ε*invNd

    for it in 1:maxiter
        maxerr = zero(F)
        @inbounds for ax in 1:d, t in 1:n
            r = _sliceview(h, ax, t)
            S = sum(r)
            if S > 0
                α = invN / S
                @. r *= α
                maxerr = max(maxerr, abs(S - invN))
            else
                fill!(r, invN/(n^(d-1)))
                maxerr = max(maxerr, invN)
            end
        end
        if maxerr ≤ atol
            return h
        end
    end
    h
end