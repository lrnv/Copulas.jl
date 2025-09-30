"""
    BernsteinCopula{d}

Fields:
- `m::NTuple{d,Int}` - polynomial degrees (smoothing parameters)
- `weights::Array{Float64, d}` - precomputed grid of box measures

Constructor

    BernsteinCopula(C; m=10)
    BernsteinCopula(data; m=10)

The Bernstein copula in dimension ``d`` is defined as

``
B_m(C)(u) = \\sum_{s_1=0}^{m_1} \\cdots \\sum_{s_d=0}^{m_d}C\\left(\\tfrac{s_1}{m_1}, \\ldots, \\tfrac{s_d}{m_d}\\right)\\prod_{j=1}^d \\binom{m_j}{s_j} u_j^{s_j}(1-u_j)^{m_j-s_j}.
``

It is a polynomial approximation of the base copula ``C`` using the multivariate Bernstein operator.

**Implementation notes:**
- The grid of box measures (weights) is fully precomputed and stored as an ``n``-dimensional array at construction. This enables fast evaluation of the copula and its density, but can be memory-intensive for large ``d`` or ``m``.
- The choice of `m` controls the smoothness of the approximation: larger `m` yields finer approximation but exponentially increases memory and computation cost (``\\prod_j m_j`` boxes).
- For high dimensions or large ``m``, memory usage may become prohibitive; see documentation for scaling behavior.
- If ``C`` is an `EmpiricalCopula`, the constructor produces the *empirical Bernstein copula*, a smoothed version of the empirical copula.
- Supports `cdf`, `logpdf`, and random generation via mixtures of beta distributions.

References:
* [sancetta2004bernstein](@cite) Sancetta, A., & Satchell, S. (2004). The Bernstein copula and its applications to modeling and approximations of multivariate distributions. Econometric Theory, 20(3), 535-562.
* [segers2017empirical](@cite) Segers, J., Sibuya, M., & Tsukahara, H. (2017). The empirical beta copula. Journal of Multivariate Analysis, 155, 35-51.
"""
struct BernsteinCopula{d} <: Copula{d}
    m::NTuple{d,Int}
    weights::Array{Float64, d}
    function BernsteinCopula(base::Copula; m::Union{Int,Tuple,Nothing}=10)
        d = Copulas.length(base)
        mtuple = nothing
        if m !== nothing
            mtuple = (m isa Int) ? ntuple(_->m, d) : m
            @assert length(mtuple) == d "The parameter m must have length $d"
            if base isa EmpiricalCopula
                n = size(base.u, 2)
                for mj in mtuple
                    if n % mj != 0
                        @warn "Sample size n=$n is not a multiple of m=$mj; partition may be unbalanced."
                    end
                end
            end
        elseif base isa EmpiricalCopula
            n = size(base.u, 2)
            m_est = max(2, floor(Int, n^(1/d)))
            @info "Automatic choice: m=$m_est in each dimension (≈ n^(1/d))."
            mtuple = ntuple(_->m_est, d)
        else
            mtuple = ntuple(_->10, d)
        end
        # Precompute CDF grid
        weights = Array{Float64}(undef, (mi+1 for mi in mtuple)...)
        for idx in CartesianIndices(weights)
            u = ntuple(j -> (idx[j]-1) / mtuple[j], d)
            weights[idx] = Distributions.cdf(base, collect(u))
        end
        # Compute measures values using multidimensional finite differences
        for axis in 1:d
            weights = Base.diff(weights, dims=axis)
        end
        return new{d}(mtuple, weights)
    end
    BernsteinCopula{d}(m::NTuple{d, Int}, weights::Array{Float64, d}) where d = new{d}(m, weights) # cheating constructor. 
end
BernsteinCopula(data::AbstractMatrix; m::Union{Int,Tuple,Nothing}=nothing, pseudo_values=true) = BernsteinCopula(EmpiricalCopula(data; pseudo_values=pseudo_values); m=m)

@inline function _bernvec_all(u::T, m::Int) where {T<:Real}
    v = zeros(T, m+1)
    if iszero(u)
        v[1] = 1; return v
    elseif isone(u)
        v[end] = 1; return v
    end
    inv1mu = 1 - u
    r = u / inv1mu
    p = inv1mu^m
    v[1] = p
    @inbounds for s in 1:m
        p *= ((m - s + 1) / s) * r
        v[s+1] = p
    end
    return v
end
@inline function _betavec_pdf_all(u::T, m::Int) where {T<:Real}
    v = zeros(T, m)
    if iszero(u)
        v[1] = m; return v
    elseif isone(u)
        v[m] = m; return v
    end
    inv1mu = 1 - u
    r = u / inv1mu
    q = inv1mu^(m-1)
    v[1] = q
    @inbounds for s in 1:m-1   # s = k+1, k=0..m-2
        q *= ((m - s) / s) * r
        v[s+1] = q
    end
    return v .* m
end
function _cdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    m = B.m
    P = ntuple(j -> _bernvec_all(u[j], m[j]), d)
    total = zero(eltype(first(P)))
    @inbounds for s in Iterators.product((0:mi for mi in m)...)
        w = 0.0
        for t in Iterators.product((1:s[j] for j in 1:d)...)
            w += B.weights[t...]
        end
        iszero(w) && continue
        total += w * prod(P[j][s[j]+1] for j in 1:d)
    end
    return total
end

function Distributions._logpdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    m = B.m
    BetaV = ntuple(j -> _betavec_pdf_all(u[j], m[j]), d)
    dens = zero(eltype(first(BetaV)))
    weights = B.weights
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        w = weights[(s[j]+1 for j in 1:d)...]
        iszero(w) && continue
        dens += w * prod(BetaV[j][s[j]+1] for j in 1:d)
    end
    return min(log(dens), zero(dens))
end

function Distributions._rand!(rng::Distributions.AbstractRNG, B::BernsteinCopula{d}, u::AbstractVector{T}) where {d,T<:Real}
    m = B.m
    target = rand(rng)
    cum = 0.0
    picked = nothing
    weights = B.weights
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        w = weights[(s[j]+1 for j in 1:d)...]
        w <= 0 && continue
        cum += w
        if cum >= target
            picked = s
            break
        end
    end
    s = picked === nothing ? ntuple(j -> m[j]-1, d) : picked
    @inbounds for j in 1:d
        u[j] = Distributions.rand(rng, Distributions.Beta(s[j] + 1, m[j] - s[j]))
    end
    return u
end

function DistortionFromCop(B::BernsteinCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,p}
    # Build mixture weights over s_i given fixed u_J for J = js.
    # Return a MixtureModel(Beta...) directly (Distortion call will push-forward marginals).
    Iset = Tuple(setdiff(1:D, js))
    @assert i in Iset "i must refer to a non-conditioned coordinate"
    m = B.m
    mi = m[i]
    α = zeros(Float64, mi)
    # Iterate over s on the grid
    for s in Iterators.product((0:(mj-1) for mj in m)...)
        wJ = 1.0
        @inbounds for (t, j) in pairs(js)
            wJ *= Distributions.pdf(Distributions.Beta(s[j] + 1, m[j] - s[j]), uⱼₛ[t])
            wJ == 0.0 && break
        end
        wJ == 0.0 && continue
        Δ = B.weights[(s[j]+1 for j in 1:D)...]
        (Δ <= 0) && continue
        α[s[i] + 1] += Δ * wJ
    end
    sα = sum(α)
    if sα <= 0
        return NoDistortion()
    end
    α ./= sα
    comps = [Distributions.Beta(k, mi - (k - 1)) for k in 1:mi]
    return Distributions.MixtureModel(comps, α)
end

# Fitting colocated. 
StatsBase.dof(::BernsteinCopula) = 0
_available_fitting_methods(::Type{<:BernsteinCopula}) = (:bernstein,)
"""
    _fit(::Type{<:BernsteinCopula}, U, ::Val{:bernstein};
         m::Union{Int,Tuple,Nothing}=nothing, pseudo_values::Bool=true, kwargs...) -> (C, meta)

Empirical plug-in fitting of `BernsteinCopula` based on `U`, using the empirical copula and (optionally) a degree `m` per dimension.

# Arguments
- `U::AbstractMatrix`: `d×n` pseudo-observations (if `pseudo_values=true`) or raw data.
- `m`: integer (same degree in all coordinates), tuple of degrees per dimension,
or `nothing` for automatic selection.
- `pseudo_values`: if `false`, pseudo-observations are constructed with `pseudos(U)`.
- `kwargs...`: forwarded to the `BernsteinCopula` constructor.

# Returns
- `(C, meta)` where `C::BernsteinCopula` and
`meta = (; emp_kind = :bernstein, pseudo_values, m = C.m)`.

**Note**: Method with no free parameters (`dof=0`).
"""
function _fit(::Type{<:BernsteinCopula}, U, ::Val{:bernstein};
              m::Union{Int,Tuple,Nothing}=nothing,
              pseudo_values::Bool=true, kwargs...)
    C = BernsteinCopula(U; m=m, pseudo_values=pseudo_values, kwargs...)
    return C, (; emp_kind=:bernstein, pseudo_values, m=C.m)
end

function SubsetCopula(C::BernsteinCopula{d}, dims::NTuple{p, Int}) where {d,p}
    # dims: indices to keep, e.g. (1,3) for a 3D copula
    # Step 1: Permute axes so that kept dims are first
    all_axes = collect(1:d)
    sum_axes = setdiff(all_axes, collect(dims))
    perm = vcat(collect(dims), sum_axes)
    permuted_weights = PermutedDimsArray(C.weights, perm)
    # Step 2: Sum over trailing axes (those not in dims)
    new_m = ntuple(i -> C.m[dims[i]], p)
    to_sum_and_drop = tuple(i for i in p+1:d)
    return BernsteinCopula{p}(new_m, dropdims(sum(permuted_weights, dims=to_sum_and_drop), dims=to_sum_and_drop))
end