"""
    CheckerboardCopula{d, T}

Fields:
- `m::Vector{Int}` — length d; number of partitions per dimension (grid resolution).
- `boxes::Dict{NTuple{d,Int}, T}` — dictionary-like mapping from grid box indices to empirical weights.
    Typically `Dict{NTuple{d,Int}, Float64}` built with `StatsBase.proportionmap`.

Constructor:

    CheckerboardCopula(X; m=nothing, pseudo_values=true)

Builds a piecewise-constant (histogram) copula on a regular grid. The unit cube
in each dimension i is partitioned into `m[i]` equal bins. Each observation is
assigned to a box `k ∈ ∏_i {0, …, m[i]-1}`; the empirical box weights `w_k`
sum to 1. The copula density is constant inside each box, with

    c(u) = w_k × ∏_i m[i]  when u ∈ box k,  and  0 otherwise.

The CDF admits the multilinear overlap form

    C(u) = ∑_k w_k × ∏_i clamp(m[i]·u_i − k_i, 0, 1),

which this type evaluates directly without storing all grid corners.

Notes:

- If `m` is `nothing`, we use `m = fill(n, d)` where `n = size(X, 2)`.
- When `pseudo_values=true` (default), `X` must already be pseudo-observations
  in [0,1]. Otherwise pass raw data and set `pseudo_values=false` to convert
  via `pseudos(X)`.
- Each `m[i]` must divide `n` to produce a valid checkerboard on the sample grid;
  this is enforced by the constructor.

References
* Neslehova (2007). *On rank correlation measures for non-continuous random variables*.
* Durante, Sanchez & Sempi (2013) *Multivariate patchwork copulas: a unified approach with applications to partial comonotonicity*.
* Segers, Sibuya & Tsukahara (2017). *The empirical beta copula*. J. Multivariate Analysis, 155, 35-51.
* Genest, Neslehova & Rémillard (2017) *Asymptotic behavior of the empirical multilinear copula process under broad conditions*.
* Cuberos, Masiello & Maume-Deschamps (2019) *Copulas checker-type approximations: application to quantiles estimation of aggregated variables*.
* Fredricks & Hofert (2025). *On the checkerboard copula and maximum entropy*.
"""
struct CheckerboardCopula{d, T} <: Copula{d}
    m::Vector{Int}
    boxes::Dict{NTuple{d,Int}, T}
end
function CheckerboardCopula(X::AbstractMatrix{T}; m=nothing, pseudo_values::Bool=true) where T
    d,n = size(X)
    ms = isnothing(m) ? fill(n,d) : m isa Integer ? fill(Int(m), d) : m
    @assert length(ms) == d && all(ms .% n .== 0) "You provided m=$m to the Checkerboard constructor, while you need to provide an integer dividing n=$n or a vector of d=$d integers, all dividing n=$n."
    # Map samples to integer box indices in each dimension (clamp right edge into m_i-1)
    data = min.(ms .- 1, floor.(Int, (pseudo_values ? X : pseudos(X)) .* ms))
    # Build a dictionary of box proportions using tuple keys
    keys_iter = (Tuple(@view data[:, j]) for j in 1:n)
    boxes = StatsBase.proportionmap(collect(keys_iter))
    return CheckerboardCopula{d, eltype(values(boxes))}(ms, boxes)
end

function Distributions.pdf(C::CheckerboardCopula{d}, u) where {d}
    # the goal is to find the right box. 
    b = Tuple(min.(C.m .- 1, floor.(Int, u .* C.m)))
    return haskey(C.boxes, b) ? C.boxes[b] * prod(C.m) : 0.0
end
function _cdf(C::CheckerboardCopula{d}, u) where {d}
    um = u .* C.m
    # Histogram/overlap CDF: sum over boxes of w_k × ∏_i clamp(m_i u_i − k_i, 0, 1)
    return sum(w * prod(clamp.(um .- box, 0, 1)) for (box, w) in C.boxes)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CheckerboardCopula{d}, u::AbstractVector{T}) where {d,T<:Real}
    # Draw a box index according to weights
    r = rand(rng)
    acc = 0.0
    chosen = nothing
    @inbounds for (box, w) in C.boxes
        acc += w
        if r <= acc
            chosen = box
            break
        end
    end
    # Fallback in case of tiny numerical drift (select the last box)
    if chosen === nothing
        for (box, _) in C.boxes
            chosen = box
        end
    end
    # Sample uniformly inside the chosen box
    @inbounds for i in 1:d
        bi = chosen[i]  # works for Tuple or Vector keys
        u[i] = T((bi + rand(rng)) / C.m[i])
    end
    return u
end

@inline function DistortionFromCop(C::CheckerboardCopula{D,T}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, i::Int) where {D,T}
    # p = 1 case; compute conditional marginal for axis i given U_j = u_j
    j = js[1]
    # Locate J bin index
    kⱼ = min(C.m[j]-1, floor(Int, C.m[j] * uⱼₛ[1]))
    # Aggregate weights over i-bins where J-index matches
    mᵢ = C.m[i]
    α = zeros(Float64, mᵢ)
    for (box, w) in C.boxes
        box[j] == kⱼ || continue
        ki = box[i]
        α[ki+1] += w
    end
    s = sum(α)
    if s <= 0
        # Degenerate slice (no box observed at this J index): fall back to uniform
        fill!(α, 1.0/mᵢ)
    else
        α ./= s
    end
    return HistogramBinDistortion(mᵢ, α)
end

@inline function ConditionalCopula(C::CheckerboardCopula{D,T}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D,T,p}
    # Project boxes onto remaining axes with J-bin fixed by uⱼₛ
    J = collect(js)
    I = collect(setdiff(1:D, J))
    # Compute J-bin indices for the conditioning point
    kJ = ntuple(t -> min(C.m[J[t]]-1, floor(Int, C.m[J[t]] * uⱼₛ[t])), p)
    # Aggregate weights for projected I-box keys
    proj = Dict{NTuple{length(I),Int}, Float64}()
    for (box, w) in C.boxes
        match = true
        @inbounds for t in 1:p
            if box[J[t]] != kJ[t]
                match = false; break
            end
        end
        match || continue
        keyI = ntuple(r -> box[I[r]], length(I))
        proj[keyI] = get(proj, keyI, 0.0) + w
    end
    # Normalize
    s = sum(values(proj))
    if s > 0
        for k in keys(proj); proj[k] /= s; end
    else
        # No matching boxes: return independent uniform on remaining dims
        proj = Dict(ntuple(r -> 0, length(I)) => 1.0)
    end
    mI = C.m[I]
    return CheckerboardCopula{length(I), Float64}(mI, proj)
end

# Fit API: mirror constructor for the moment until we get a better API ?
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
function Distributions.fit(::Type{CT}, u; m=nothing, pseudo_values::Bool=true) where {CT<:CheckerboardCopula}
    return CheckerboardCopula(u; m=m, pseudo_values=pseudo_values)
end