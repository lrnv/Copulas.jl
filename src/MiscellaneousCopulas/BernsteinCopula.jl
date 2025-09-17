"""
    BernsteinCopula{d, C}

Fields:
- `base::C` - underlying copula
- `m::NTuple{d,Int}` - polynomial degrees (smoothing parameters)

Constructor

    BernsteinCopula(C; m=10)
    BernsteinCopula(data; m=10)

The Bernstein copula in dimension ``d`` is defined as

```math
B_m(C)(u) = \\sum_{s_1=0}^{m_1} \\cdots \\sum_{s_d=0}^{m_d}C\\left(\\tfrac{s_1}{m_1}, \\ldots, \\tfrac{s_d}{m_d}\\right)\\prod_{j=1}^d \\binom{m_j}{s_j} u_j^{s_j}(1-u_j)^{m_j-s_j}.
```

It is a polynomial approximation of the base copula ``C`` using the multivariate Bernstein operator.

Notes:

- If ``C`` is an `EmpiricalCopula`, the constructor produces the *empirical Bernstein copula*, a smoothed version of the empirical copula.
- Supports `cdf`, `logpdf` and random generation via mixtures of beta distributions.
- The choice of `m` controls the smoothness of the approximation: larger `m` yields finer approximation but higher computational cost.

References:
* [sancetta2004bernstein](@cite) Sancetta, A., & Satchell, S. (2004). The Bernstein copula and its applications to modeling and approximations of multivariate distributions. Econometric Theory, 20(3), 535-562.
* [segers2017empirical](@cite) Segers, J., Sibuya, M., & Tsukahara, H. (2017). The empirical beta copula. Journal of Multivariate Analysis, 155, 35-51.
"""
struct BernsteinCopula{d,C<:Copula} <: Copula{d}
    base::C
    m::NTuple{d,Int}
    function BernsteinCopula(base::Copula; m::Union{Int,Tuple,Nothing}=10)
        d = Copulas.length(base)

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
            return new{d,typeof(base)}(base, mtuple)
        end

        if base isa EmpiricalCopula
            n = size(base.u, 2)
            m_est = max(2, floor(Int, n^(1/d)))
            @info "Automatic choice: m=$m_est in each dimension (≈ n^(1/d))."
            return new{d,typeof(base)}(base, ntuple(_->m_est, d))
        end

        return new{d,typeof(base)}(base, ntuple(_->10, d))
    end
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
        w = Distributions.cdf(B.base, s ./ m)
        iszero(w) && continue
        total += w * prod(P[j][s[j]+1] for j in 1:d)
    end
    return total
end

function Distributions._logpdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    m = B.m
    BetaV = ntuple(j -> _betavec_pdf_all(u[j], m[j]), d)
    dens = zero(eltype(first(BetaV)))
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        w = measure(B.base, ntuple(j -> s[j] / m[j], d), ntuple(j -> (s[j] + 1) / m[j], d))
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
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        w = measure(B.base, ntuple(j -> s[j] / m[j], d), ntuple(j -> (s[j] + 1) / m[j], d))
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
        Δ = measure(B.base, ntuple(j -> s[j] / m[j], D), ntuple(j -> (s[j] + 1) / m[j], D))
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