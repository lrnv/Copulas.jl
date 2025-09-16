"""
    BernsteinCopula{d, C}

Fields:
- `base::C` - underlying copula
- `m::NTuple{d,Int}` - polynomial degrees (smoothing parameters)

Constructor

    BernsteinCopula(C; m=10)
    BernsteinCopula(data; m=10)
    EmpiricalBernsteinCopula(data; m=10)

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
end

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
        return BernsteinCopula{d,typeof(base)}(base, mtuple)
    end

    if base isa Copulas.EmpiricalCopula
        n = size(base.u, 2)
        m_est = max(2, floor(Int, n^(1/d)))
        @info "Automatic choice: m=$m_est in each dimension (≈ n^(1/d))."
        return BernsteinCopula{d,typeof(base)}(base, ntuple(_->m_est, d))
    end

    return BernsteinCopula{d,typeof(base)}(base, ntuple(_->10, d))
end

@inline function DistortionFromCop(B::BernsteinCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,p}
    # Build mixture weights over s_i given fixed u_J for J = js.
    # Return a MixtureModel(Beta...) directly (Distortion call will push-forward marginals).
    Iset = Tuple(setdiff(1:D, js))
    @assert i in Iset "i must refer to a non-conditioned coordinate"
    m = B.m
    mi = m[i]
    α = zeros(Float64, mi)
    # Iterate over s on the grid
    for s in Iterators.product((0:(mj-1) for mj in m)...)
        # contribution weight for conditioned coordinates
        wJ = 1.0
        @inbounds for (t, j) in pairs(js)
            a = s[j] + 1
            b = m[j] - s[j]
            wJ *= Distributions.pdf(Distributions.Beta(a, b), uⱼₛ[t])
        end
        wJ == 0.0 && continue
        Δ = delta_d(B.base, s, m)
        if Δ > 0
            α[s[i] + 1] += Δ * wJ
        end
    end
    sα = sum(α)
    if sα <= 0
        return NoDistortion()
    end
    α ./= sα
    comps = [Distributions.Beta(k, mi - (k - 1)) for k in 1:mi]
    return Distributions.MixtureModel(comps, α)
end

function BernsteinCopula(data::AbstractMatrix; m::Union{Int,Tuple,Nothing}=nothing)
    EC = Copulas.EmpiricalCopula(data; pseudo_values=false)
    return BernsteinCopula(EC; m=m)
end

EmpiricalBernsteinCopula(data::AbstractMatrix; m::Union{Int,Tuple,Nothing}=nothing) = BernsteinCopula(data; m=m)

function delta_d(C::Copula, s::NTuple{d,Int}, m::NTuple{d,Int}) where {d}
    total = zero(Float64)
    @inbounds for ε in Iterators.product((0:1 for _ in 1:d)...)
        u = [ (s[j] + ε[j]) / m[j] for j in 1:d ]
        sign = (-1)^(d - sum(ε))
        total += sign * Distributions.cdf(C, u)
    end
    return total
end

@inline function _bernvec_all(u::T, m::Int) where {T<:Real}
    v = Vector{T}(undef, m+1)
    if u == zero(T)
        v[1] = one(T);  @inbounds for s in 2:m+1 v[s]=zero(T) end
        return v
    elseif u == one(T)
        @inbounds for s in 1:m v[s]=zero(T) end; v[end]=one(T)
        return v
    end
    one_minus_u = one(T) - u
    p = one_minus_u^m
    v[1] = p
    @inbounds for s in 0:m-1
        p *= ( (m - s) / (s + 1) ) * ( u / one_minus_u )
        v[s+2] = p
    end
    return v
end

@inline function _betavec_pdf_all(u::T, m::Int) where {T<:Real}
    v = Vector{T}(undef, m)
    if u == zero(T)
        v[1] = T(m); @inbounds for s in 2:m v[s]=zero(T) end
        return v
    elseif u == one(T)
        @inbounds for s in 1:m-1 v[s]=zero(T) end; v[m]=T(m)
        return v
    end
    one_minus_u = one(T) - u
    q = one_minus_u^(m-1)
    v[1] = T(m) * q
    @inbounds for s in 0:m-2
        q *= ( (m - 1 - s) / (s + 1) ) * ( u / one_minus_u )
        v[s+2] = T(m) * q
    end
    return v
end

function Distributions.cdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    @assert length(u) == d
    m = B.m

    P = ntuple(j -> _bernvec_all(eltype(u)(u[j]), m[j]), d)
    total = zero(eltype(first(P)))
    @inbounds for s in Iterators.product((0:mi for mi in m)...)
        coeff = Distributions.cdf(B.base, [s[j]/m[j] for j in 1:d])
        basis = one(coeff)
        @inbounds for j in 1:d
            basis *= P[j][s[j]+1]
        end
        total += coeff * basis
    end
    return total
end

function Distributions.logpdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    @assert length(u) == d
    m = B.m
    BetaV = ntuple(j -> _betavec_pdf_all(eltype(u)(u[j]), m[j]), d)
    dens = zero(eltype(first(BetaV)))
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        ΔC = delta_d(B.base, s, m)
        if ΔC != 0.0
            prodβ = one(ΔC)
            @inbounds for j in 1:d
                prodβ *= BetaV[j][s[j]+1]
            end
            dens += ΔC * prodβ
        end
    end
    dens = max(dens, zero(dens))
    return log(dens + eps(Float64))
end

function Distributions._rand!(rng::Distributions.AbstractRNG, B::BernsteinCopula{d}, u::AbstractVector{T}) where {d,T<:Real}
    m = B.m
    n = prod(mi for mi in m)

    keys_ = Vector{NTuple{d,Int}}(undef, n)
    vals_ = Vector{Float64}(undef, n)

    k = 1
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        keys_[k] = s
        vals_[k] = delta_d(B.base, s, m)
        k += 1
    end
    @inbounds for i in eachindex(vals_)
        if (vals_[i] < 0.0) && (vals_[i] > -sqrt(eps(Float64)))
            vals_[i] = 0.0
        end
    end
    S = sum(vals_)
    S <= 0 && error("No positive mass in Δ; check base copula or m.")
    vals_ ./= S

    s = StatsBase.sample(rng, keys_, StatsBase.Weights(vals_))

    @inbounds for j in 1:d
        u[j] = Distributions.rand(rng, Distributions.Beta(s[j]+1, m[j]-s[j]))
    end
    return u
end

function Distributions._rand!(rng::Distributions.AbstractRNG, B::BernsteinCopula{d}, U::AbstractMatrix{T}) where {d,T<:Real}
    # Fill columns of U with i.i.d. samples; precompute s-weights once
    @assert size(U, 1) == d
    m = B.m
    nkeys = prod(mi for mi in m)
    keys_ = Vector{NTuple{d,Int}}(undef, nkeys)
    vals_ = Vector{Float64}(undef, nkeys)
    k = 1
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        keys_[k] = s
        vals_[k] = delta_d(B.base, s, m)
        k += 1
    end
    # Clip small negatives from numerical noise
    @inbounds for i in eachindex(vals_)
        if (vals_[i] < 0.0) && (vals_[i] > -sqrt(eps(Float64)))
            vals_[i] = 0.0
        end
    end
    S = sum(vals_)
    S <= 0 && error("No positive mass in Δ; check base copula or m.")
    vals_ ./= S
    ws = StatsBase.Weights(vals_)

    @inbounds for jcol in axes(U, 2)
        s = StatsBase.sample(rng, keys_, ws)
        for j in 1:d
            U[j, jcol] = Distributions.rand(rng, Distributions.Beta(s[j]+1, m[j]-s[j]))
        end
    end
    return U
end

function _gridC_2d(base::Copula, m1::Int, m2::Int)
    Tret = typeof(Distributions.cdf(base, [0.0, 0.0]))
    M = Matrix{Tret}(undef, m1+1, m2+1)
    @inbounds for s in 0:m1, t in 0:m2
        M[s+1, t+1] = Distributions.cdf(base, [s/m1, t/m2])
    end
    return M
end

function _delta_2d(base::Copula, m1::Int, m2::Int)
    Tret = typeof(Distributions.cdf(base, [0.0, 0.0]))
    Δ = Matrix{Tret}(undef, m1, m2)
    @inbounds for s in 0:m1-1, t in 0:m2-1
        c11 = Distributions.cdf(base, [(s+1)/m1, (t+1)/m2])
        c10 = Distributions.cdf(base, [(s+1)/m1, t/m2])
        c01 = Distributions.cdf(base, [s/m1, (t+1)/m2])
        c00 = Distributions.cdf(base, [s/m1, t/m2])
        Δ[s+1, t+1] = c11 - c10 - c01 + c00
    end
    if Tret <: Real
        w = vec(Δ)
        @inbounds for k in eachindex(w)
            if (w[k] < 0.0) && (w[k] > -sqrt(eps(Float64)))
                w[k] = zero(Tret)
            end
        end
    end
    return Δ
end

function Distributions.cdf(B::BernsteinCopula{2}, u::AbstractVector)
    @assert length(u) == 2
    m1, m2 = B.m
    bu = _bernvec_all(eltype(u)(u[1]), m1)
    bv = _bernvec_all(eltype(u)(u[2]), m2)
    M  = _gridC_2d(B.base, m1, m2)
    t  = M * bv
    return LinearAlgebra.dot(bu, t)
end

function Distributions.logpdf(B::BernsteinCopula{2}, u::AbstractVector)
    @assert length(u) == 2
    m1, m2 = B.m
    βu = _betavec_pdf_all(eltype(u)(u[1]), m1)
    βv = _betavec_pdf_all(eltype(u)(u[2]), m2)
    Δ  = _delta_2d(B.base, m1, m2)
    t  = Δ * βv
    dens = LinearAlgebra.dot(βu, t)
    dens = max(dens, zero(dens))
    return log(dens + eps(Float64))
end
