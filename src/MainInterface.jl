# Three small utility functions. 

@inline _δ(t) = oftype(t, 1e-12)
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))
_invmono(f; tol=1e-8, θmax=1e6, a=0.0, b=1.0) = begin
    fa,fb = f(0.0), f(1.0)
    while fb ≤ 0 && b < θmax
        b = min(2b, θmax); fb = f(b)
        !isfinite(fb) && (b = θmax; break)
    end
    (fa < 0 && fb > 0) || error("Could not bound root at [0, $θmax].")
    Roots.find_zero(f, (a,b), Roots.Brent(); atol=tol, rtol=tol)
end

###############################################################################
#####  Main Copula interface. 
#####  User-facing function: 
#####       1) Distributions.jl's API: cdf, pdf, logpdf, loglikelyhood, etc..
#####       2) ρ, τ, β, γ, η, λₗ and λᵤ: repectively the spearman rho, kendall tau,  blomqvist's beta, 
#####          gini's gamma,entropy eta, and lower and upper tail dependencies. 
#####       3) measure(C, us, vs) that get the measure associated with the copula.  
#####       3) pseudo(data) construct pseudo-data from a given dataset.  
#####
#####  When implementing a new copula, you have to overwrite `Copulas._cdf()`
#####  and you may overwrite ρ, τ, β, γ, η, λₗ, λᵤ, measure for performances. 
###############################################################################
abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.broadcastable(C::Copula) = Ref(C)
Base.length(::Copula{d}) where d = d
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
function _cdf(C::CT,u) where {CT<:Copula}
    f(x) = Distributions.pdf(C,x)
    z = zeros(eltype(u),length(C))
    return HCubature.hcubature(f,z,u,rtol=sqrt(eps()))[1]
end

# Multivariate dependence metrics 
function ρ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    z = zeros(d)
    i = ones(d)
    r = HCubature.hcubature(F, z, i, rtol=sqrt(eps()))[1]
    return (2^d * (d+1) * r - d - 1)/(2^d - d - 1) # Ok for multivariate. 
end
function τ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    r = Distributions.expectation(F, C; nsamples=10^4)
    return (2^d / (2^(d-1) - 1)) * r - 1 / (2^(d-1) - 1)
end
function β(C::Copula{d}) where {d}
    d == 2 && return 4*Distributions.cdf(C, [0.5, 0.5]) - 1
    u     = fill(0.5, d)
    C0    = Distributions.cdf(C, u)
    Cbar0 = Distributions.cdf(SurvivalCopula(C, Tuple(1:d)), u)
    return (2.0^(d-1) * C0 + Cbar0 - 1) / (2^(d-1) - 1)
end
function γ(C::Copula{d}; nmc::Int=100_000, rng::Random.AbstractRNG=Random.MersenneTwister(123)) where {d}
    d ≥ 2 || throw(ArgumentError("γ(C) requires d≥2"))
    if d == 2
        f(t) = Distributions.cdf(C, [t, t]) + Distributions.cdf(C, [t, 1 - t])
        I, _ = QuadGK.quadgk(f, 0.0, 1.0; rtol=sqrt(eps()))
        return -2 + 4I
    end
    @inline _A(u)    = (minimum(u) + max(sum(u) - d + 1, 0.0)) / 2
    @inline _Abar(u) = (1 - maximum(u) + max(1 - sum(u), 0.0)) / 2
    @inline invfac(k::Integer) = exp(-SpecialFunctions.logfactorial(k))
    s = 0.0
    @inbounds for i in 0:d
        s += (isodd(i) ? -1.0 : 1.0) * binomial(d, i) * invfac(i + 1)
    end
    a_d = 1/(d + 1) + 0.5*invfac(d + 1) + 0.5*s
    b_d = 2/3 + 4.0^(1 - d) / 3
    U = rand(rng, C, nmc)
    m = 0.0
    @inbounds for j in 1:nmc
        u = @view U[:, j]
        m += _A(u) + _Abar(u)
    end
    m /= nmc
    return (m - a_d) / (b_d - a_d)
end
function η(C::Copula{d}; nmc::Int=100_000, rng::Random.AbstractRNG=Random.MersenneTwister(123)) where {d}
    U = rand(rng, C, nmc)
    s = 0.0
    @inbounds for j in 1:nmc
        u  = @view U[:, j]
        lp = Distributions.logpdf(C, u)
        isfinite(lp) || throw(DomainError(lp, "logpdf(C,u) non-finite."))
        s -= lp
    end
    H = s / nmc
    t = clamp(2H, -700.0, 0.0)
    r = sqrt(max(0.0, 1 - exp(t)))
    return (H = H, I = -H, r = r)
end
function λₗ(C::Copula{d}; ε::Float64 = 1e-10) where {d} 
    g(e) = Distributions.cdf(C, fill(e, d)) / e
    return clamp(2*g(ε/2) - g(ε), 0.0, 1.0)
end
function λᵤ(C::Copula{d}; ε::Float64 = 1e-10) where {d} 
    Sc   = SurvivalCopula(C, Tuple(1:d))
    f(e) = Distributions.cdf(Sc, fill(e, d)) / e
    return clamp(2*f(ε/2) - f(ε), 0.0, 1.0)
end

# Multivariate dependence metrics applied to a matrix. 
function β(U::AbstractMatrix)
    # Assumes psuedo-data given. β multivariate (Hofert–Mächler–McNeil, ec. (7))
    d, n = size(U)
    count = sum(j -> all(U[:, j] .<= 0.5) || all(U[:, j] .> 0.5), 1:n)
    h_d = 2.0^(d-1) / (2.0^(d-1) - 1.0)
    return h_d * (count/n - 2.0^(1-d))
end
function τ(U::AbstractMatrix)
    # Sample version of multivariate Kendall's tau for pseudo-data
    d, n = size(U)
    comp = 0
    @inbounds for j in 2:n, i in 1:j-1
        uᵢ = @view U[:, i]; uⱼ = @view U[:, j]
        comp += (all(uᵢ .<= uⱼ) || all(uᵢ .>= uⱼ))
    end
    pc = comp / (n*(n-1)/2)
    return (2.0^d * pc - 2.0) / (2.0^d - 2.0)
end
function ρ(U::AbstractMatrix)
    # Sample version of multivariate Spearman's tau for pseudo-data
    d, n = size(U)
    R = hcat((StatsBase.tiedrank(U[k, :]) for k in 1:d)...)   # n×d
    μ = Statistics.mean(prod(R, dims=2)) / (n + 1)^d          # ≈ E[∏ U_i]
    h = (d + 1) / (2.0^d - (d + 1))
    return h * (2.0^d * μ - 1.0)
end
function γ(U::AbstractMatrix)
    # Assumes pseudo-data given. Multivariate Gini’s gamma (Behboodian–Dolati–Úbeda, 2007)
    d, n = size(U)
    if d == 2
    # Schechtman–Yitzhaki symmetric Gini over ranks (copular invariant)
        r1 = StatsBase.tiedrank(@view U[1, :])
        r2 = StatsBase.tiedrank(@view U[2, :])
        m  = n
        h  = m + 1
        acc = 0.0
        @inbounds @simd for k in 1:m
            acc += abs(r1[k] + r2[k] - h) - abs(r1[k] - r2[k])
        end
        return 2*acc / (m*h)
    else
        @inline _A(u)    = (minimum(u) + max(sum(u) - d + 1, 0.0)) / 2
        @inline _Abar(u) = (1 - maximum(u) + max(1 - sum(u), 0.0)) / 2
        invfac(k::Integer) = exp(-SpecialFunctions.logfactorial(k))
        s = 0.0
        binomf(d,i) = exp(SpecialFunctions.loggamma(d+1) - SpecialFunctions.loggamma(i+1) - SpecialFunctions.loggamma(d-i+1))
        @inbounds for i in 0:d
            s += (isodd(i) ? -1.0 : 1.0) * binomf(d,i) * invfac(i + 1)
        end
        a_d = 1/(d + 1) + 0.5*invfac(d + 1) + 0.5*s
        b_d = 2/3 + 4.0^(1 - d) / 3
        m = 0.0
        @inbounds for j in 1:n
            u = @view U[:, j]
            m += _A(u) + _Abar(u)
        end
        m /= n
        return (m - a_d) / (b_d - a_d)
    end
end
function _λ(U::AbstractMatrix; t::Symbol=:upper, p::Union{Nothing,Real}=nothing)
    # Assumes pseudo-data given. Multivariate tail’s lambda (Schmidt, R. & Stadtmüller, U. 2006)
    d, m = size(U)
    m ≥ 4 || throw(ArgumentError("At least 4 observations are required"))
    p === nothing && (p = 1/sqrt(m))
    (0 < p < 1) || throw(ArgumentError("p must be in (0,1)"))
    V = t === :upper ? (1 .- Float64.(U)) : Float64.(U)
    cnt = 0
    @inbounds @views for j in 1:m
        cnt += all(V[:, j] .<= p)   # vista sin copiar gracias a @views
    end
    return clamp(cnt / (p*m), 0.0, 1.0)
end
λₗ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:lower, p=p)
λᵤ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:upper, p=p)
function η(U::AbstractMatrix; k::Int=5, p::Real=Inf, leafsize::Int=32)
    # Assumes pseudo-data given. Multivariate copula entropy (L.F. Kozachenko and N.N. Leonenko., 1987)
    d, n = size(U)
    n ≥ k+1 || throw(ArgumentError("n ≥ k+1 is required"))
    (p ≥ 1 || isinf(p)) || throw(ArgumentError("invalid Minkowski norm: p ∈ [1,∞]"))
    any(isnan, U) && return NaN
    X = Array{Float64}(U)
    lp(u, v, p) = (sum(abs.(u .- v) .^ p))^(1/p)
    cheb(u, v)  = maximum(abs.(u .- v))
    function lb_lp(q, lo, hi, p)
        s = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            s += δ^p
        end
        return s^(1/p)
    end
    function lb_inf(q, lo, hi)
        m = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            m = δ > m ? δ : m
        end
        return m
    end
    nodes = Vector{Vector{Any}}()
    function build(idxs::Vector{Int})
        lo = fill( Inf, d); hi = fill(-Inf, d)
        @inbounds for j in idxs, r in 1:d
            v = X[r, j]
            lo[r] = v < lo[r] ? v : lo[r]
            hi[r] = v > hi[r] ? v : hi[r]
        end
        if length(idxs) ≤ leafsize
            push!(nodes, Any[copy(idxs), 0, 0.0, 0, 0, lo, hi])  # hoja
            return length(nodes)
        end
        spans = hi .- lo
        sd = findmax(spans)[2]
        sv = (lo[sd] + hi[sd]) / 2
        left = Int[]; right = Int[]
        @inbounds for j in idxs
            (X[sd, j] ≤ sv ? push!(left, j) : push!(right, j))
        end
        if isempty(left) || isempty(right)
            ord = sort(idxs; by = j -> X[sd, j]); m = length(ord) ÷ 2
            left  = ord[1:m]; right = ord[m+1:end]; sv = X[sd, ord[m]]
        end
        L = build(left); R = build(right)
        push!(nodes, Any[Int[], sd, sv, L, R, lo, hi])
        return length(nodes)
    end
    root = build(collect(1:n))
    function knn!(q::AbstractVector{<:Real}, selfidx::Int, K::Int,
                  D::Vector{Float64}, I::Vector{Int}, node::Int)
        nd = nodes[node]
        idxs, sd, sv, L, R, lo, hi = nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], nd[7]
        worst = isempty(D) ? Inf : maximum(D)
        lb = isinf(p) ? lb_inf(q, lo, hi) : lb_lp(q, lo, hi, p)
        if length(D) == K && lb ≥ worst
            return
        end
        if sd == 0
            @inbounds for j in idxs
                j == selfidx && continue
                xj = @view X[:, j]
                dj = isinf(p) ? cheb(q, xj) : lp(q, xj, p)
                dj = ifelse(iszero(dj), eps(Float64), dj)
                if length(D) < K
                    push!(D, dj); push!(I, j)
                elseif dj < worst
                    t = findmax(D)[2]; D[t] = dj; I[t] = j
                end
                worst = length(D) == K ? maximum(D) : Inf
            end
            return
        end
        near = (q[sd] ≤ sv) ? L : R
        far  = (q[sd] ≤ sv) ? R : L
        knn!(q, selfidx, K, D, I, near)
        worst = length(D) == K ? maximum(D) : Inf
        if length(D) < K || abs(q[sd] - sv) < worst
            knn!(q, selfidx, K, D, I, far)
        end
    end
    ρ = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        D = Float64[]; I = Int[]
        q = @view X[:, j]
        knn!(q, j, k, D, I, root)
        ρ[j] = maximum(D)
    end
    ρ .= max.(ρ, eps(Float64))

    #KL: H = -ψ(k)+ψ(n)+log c_{d,p} + (d/n)∑log ρ  ; for L∞, we absorb log c_{d,∞}=d log 2
    H = -SpecialFunctions.digamma(k) + SpecialFunctions.digamma(n)
    if isinf(p)
        H += (d / n) * sum(log.(2 .* ρ))
    else
        logcd = d*log(2*SpecialFunctions.gamma(1 + 1/p)) - SpecialFunctions.loggamma(1 + d/p)
        H += logcd + (d / n) * sum(log.(ρ))
    end
    t = clamp(2H, -700.0, 0.0)
    r = sqrt(max(0.0, 1 - exp(t)))
    return (H = H, I = -H, r = r)
end

# Measure function. 
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


"""
    pseudos(sample)

Compute the pseudo-observations of a multivariate sample. Note that the sample has to be given in wide format (d,n), where d is the dimension and n the number of observations.

Warning: the order used is ordinal ranking like https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29, see `StatsBase.ordinalrank` for the ordering we use. If you want more flexibility, checkout `NormalizeQuantiles.sampleranks`.
"""
function pseudos(sample::AbstractMatrix)
    # Fast pseudo-observations (d×n) using per-row ordinal ranks without allocations per row
    d, n = size(sample)
    U = Matrix{Float64}(undef, d, n)
    tmp_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:d
        # compute ordinal ranks for row i
        x = @view sample[i, :]
        # sortperm is stable; ordinal ranks from positions in sorted order
        sortperm!(tmp_idx, x; by=identity, alg=Base.Sort.DEFAULT_STABLE)
        # ranks: position in sorted order; ties preserve order of appearance
        for (rank, idx) in enumerate(tmp_idx)
            U[i, idx] = rank / (n + 1)
        end
    end
    return U
end

###############################################################################
#####  SklarDist framework.
#####  User-facing function: `SklarDist(C::Copula{d}, m::NTuple{d, <:UnivariateDistribution}) where d`
#####
#####  Nothing here should be overwritten when defining new copulas. 
###############################################################################

"""
    SklarDist{CT,TplMargins} 

Fields:
  - `C::CT` - The copula
  - `m::TplMargins` - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

Construct a joint distribution via Sklar's theorem from marginals and a copula. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem):

!!! theorem "Sklar 1959"
    For every random vector ``\\boldsymbol X``, there exists a copula ``C`` such that 

    ``\\forall \\boldsymbol x\\in \\mathbb R^d, F(\\boldsymbol x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).``
    The copula ``C`` is uniquely determined on ``\\mathrm{Ran}(F_{1}) \\times ... \\times \\mathrm{Ran}(F_{d})``, where ``\\mathrm{Ran}(F_i)`` denotes the range of the function ``F_i``. In particular, if all marginals are absolutely continuous, ``C`` is unique.


The resulting random vector follows the `Distributions.jl` API (rand/cdf/pdf/logpdf). A `fit` method is also provided. Example:

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset

# You may estimate a copula using the `fit` function:
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
```

References: 
* [sklar1959](@cite) Sklar, M. (1959). Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP (Vol. 8, No. 3, pp. 229-231).
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
        return new{typeof(C),typeof(m)}(C,m)
    end    
end
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
Distributions.cdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins} = Distributions.cdf(S.C,Distributions.cdf.(S.m,x))
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, x::AbstractVector{T}) where {CT,TplMargins,T}
    Random.rand!(rng,S.C,x)
     x .= Distributions.quantile.(S.m,x)
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins},u) where {CT,TplMargins}
    sum(Distributions.logpdf(S.m[i],u[i]) for i in eachindex(u)) + Distributions.logpdf(S.C,clamp.(Distributions.cdf.(S.m,u),0,1))
end
function StatsBase.dof(S::SklarDist)
    a = StatsBase.dof(S.C)
    b = sum(hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d)) for d in S.m)
    return a+b
end


###############################################################################
#####  Subsetting framework.
#####  User-facing function: `subsetdims()`
#####
#####  When implementing a new copula, you can overwrite: 
#####   - `SubsetCopula(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}`
###############################################################################
"""
    SubsetCopula{d,CT}

Fields:
  - `C::CT` - The copula
  - `dims::Tuple{Int}` - a Tuple representing which dimensions are used. 

Constructor

    SubsetCopula(C::Copula,dims)

This class allows to construct a random vector corresponding to a few dimensions of the starting copula. If ``(X_1,...,X_n)`` is the random vector corresponding to the copula `C`, this returns the copula of `(` ``X_i`` `for i in dims)`. The dependence structure is preserved. There are specialized methods for some copulas. 
"""
struct SubsetCopula{d,CT} <: Copula{d}
    C::CT
    dims::NTuple{d,Int}
    function SubsetCopula(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}
        @assert 2 <= p <= d "You cannot construct a subsetcopula with dimension p=1 or p > d (d = $d, p = $p provided)"
        dims == Tuple(1:d) && return C
        @assert all(dims .<= d)
        return new{p, typeof(C)}(C,Tuple(Int.(dims)))
    end
end
function SubsetCopula(CS::SubsetCopula{d,CT}, dims2::NTuple{p, Int}) where {d,CT,p}
    @assert 2 <= p <= d
    return SubsetCopula(CS.C, ntuple(i -> CS.dims[dims2[i]], p))
end
_available_fitting_methods(::Type{<:SubsetCopula}) = Tuple{}() # cannot be fitted. 
Base.eltype(C::SubsetCopula{d,CT}) where {d,CT} = Base.eltype(C.C)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SubsetCopula{d,CT}, x::AbstractVector{T}) where {T<:Real, d,CT}
    u = Random.rand(rng,C.C)
    x .= (u[i] for i in C.dims)
    return x
end
function _cdf(C::SubsetCopula{d,CT},u) where {d,CT}
    # Simplyu saturate dimensions that are not choosen.
    v = ones(eltype(u), length(C.C))
    for (i,j) in enumerate(C.dims)
        v[j] = u[i]
    end 
    return Distributions.cdf(C.C,v)
end
function Distributions._logpdf(S::SubsetCopula{d,<:Copula{D}}, u) where {d,D}
    return log(_partial_cdf(S.C, Tuple(setdiff(1:D, S.dims)), S.dims, ones(D-d), u))
end

# Dependence metrics are symetric in bivariate cases: 
τ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = τ(C.C)
ρ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ρ(C.C)
β(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = β(C.C)
γ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = γ(C.C)
η(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = η(C.C)
λₗ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λₗ(C.C)
λᵤ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λᵤ(C.C)

"""
    subsetdims(C::Copula, dims::NTuple{p, Int})
    subsetdims(D::SklarDist, dims)

Return a new copula or Sklar distribution corresponding to the subset of dimensions specified by `dims`.

# Arguments
- `C::Copula`: The original copula object.
- `D::SklarDist`: The original Sklar distribution.
- `dims::NTuple{p, Int}`: Tuple of indices representing the dimensions to keep.

# Returns
- A `SubsetCopula` or a new `SklarDist` object corresponding to the selected dimensions. If `p == 1`, returns a `Uniform` distribution or the corresponding marginal.

# Details
This function extracts the dependence structure among the specified dimensions from the original copula or Sklar distribution. Specialized methods exist for some copula types to ensure efficiency and correctness.
"""
function subsetdims(C::Copula{d},dims::NTuple{p, Int}) where {d,p}
    p==1 && return Distributions.Uniform()
    dims==ntuple(i->i, d) && return C
    @assert p < d
    @assert length(unique(dims))==length(dims)
    @assert all(dims .<= d)
    return SubsetCopula(C,dims)
end
function subsetdims(D::SklarDist, dims::NTuple{p, Int}) where p
    p==1 && return D.m[dims[1]]
    return SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))
end
subsetdims(C::Union{Copula, SklarDist}, dims) = subsetdims(C, Tuple(collect(Int, dims)))

# Pairwise dependence metrics, leveraging subsetting: 
function _as_biv(f::F, C::Copula{d}) where {F, d}
    K = ones(d,d)
    for i in 1:d
        for j in i+1:d
            K[i,j] = f(SubsetCopula(C, (i,j)))
            K[j,i] = K[i,j]
        end
    end
    return K
end
StatsBase.corkendall(C::Copula)  = _as_biv(τ, C)
StatsBase.corspearman(C::Copula) = _as_biv(ρ, C)
corblomqvist(C::Copula)          = _as_biv(β, C)
corgini(C::Copula)               = _as_biv(γ, C)
corentropy(C::Copula)            = _as_biv(η, C)
coruppertail(C::Copula)          = _as_biv(λᵤ, C)
corlowertail(C::Copula)          = _as_biv(λₗ, C)

# Pairwise component metrics applied to (n,d)-shaped matrices: 
function corblomqvist(X::AbstractMatrix{<:Real})
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    n = size(X, 2) 
    C = Matrix{Float64}(LinearAlgebra.I, n, n)
    anynan = Vector{Bool}(undef, n)
    m = size(X, 1)
    h = (m + 1) / 2
    for j = 1:n
        Xj = view(X, :, j)
        anynan[j] = any(isnan, Xj)
        if anynan[j]
            C[:,j] .= NaN
            C[j,:] .= NaN
            C[j,j] = 1
            continue
        end
        xrj = StatsBase.tiedrank(Xj)
        for i = 1:(j-1)
            Xi = view(X, :, i)
            if anynan[i]
                C[i,j] = C[j,i] = NaN
            else
                xri = StatsBase.tiedrank(Xi)
                c = 0
                @inbounds for k in 1:m
                    c += ( (xri[k] <= h) == (xrj[k] <= h) )
                end
                C[i,j] = C[j,i] = 2c/m - 1
            end
        end
    end
    return C
end
function corgini(X::AbstractMatrix{<:Real})
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    m, n = size(X)
    C = Matrix{Float64}(LinearAlgebra.I, n, n)
    anynan = Vector{Bool}(undef, n)
    ranks  = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        Xj = view(X, :, j)
        anynan[j] = any(isnan, Xj)
        ranks[j]  = anynan[j] ? Float64[] : StatsBase.tiedrank(Xj)
        if anynan[j]
            C[:, j] .= NaN
            C[j, :] .= NaN
            C[j, j]  = 1.0
        end
    end
    h = m + 1
    for j in 2:n
        anynan[j] && continue
        rj = ranks[j]
        for i in 1:j-1
            if anynan[i]
                C[i, j] = C[j, i] = NaN
            else
                ri  = ranks[i]
                acc = 0.0
                @inbounds @simd for k in 1:m
                    acc += abs(ri[k] + rj[k] - h) - abs(ri[k] - rj[k])
                end
                C[i, j] = C[j, i] = 2*acc / (m*h)
            end
        end
    end
    return C
end
function corentropy(X::AbstractMatrix{<:Real}; k::Int=5, p::Real=Inf, leafsize::Int=32, signed::Bool=false)
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    m, n = size(X)
    Cnan = Vector{Bool}(undef, n)
    for j in 1:n
        Cnan[j] = any(isnan, @view X[:, j])
    end
    Ucol = [Cnan[j] ? Float64[] : collect(@view X[:, j]) for j in 1:n]
    H  = zeros(Float64, n, n); I  = zeros(Float64, n, n); R  = Matrix{Float64}(LinearAlgebra.I, n, n)
    R[LinearAlgebra.diagind(R)] .= 1.0
    Rsg = nothing
    Tτ  = nothing
    if signed
        Rsg = Matrix{Float64}(LinearAlgebra.I, n, n); Rsg[LinearAlgebra.diagind(Rsg)] .= 1.0
        Tτ  = Matrix{Float64}(LinearAlgebra.I, n, n)
    end
    Ub = Array{Float64}(undef, 2, m)
    @inbounds for j in 2:n
        if Cnan[j]
            H[:, j] .= NaN; H[j, :] .= NaN; H[j, j] = 0.0
            I[:, j] .= NaN; I[j, :] .= NaN; I[j, j] = 0.0
            R[:, j] .= NaN; R[j, :] .= NaN; R[j, j] = 1.0
            if signed
                Rsg[:, j] .= NaN; Rsg[j, :] .= NaN; Rsg[j, j] = 1.0
                Tτ[:, j]  .= NaN; Tτ[j, :]  .= NaN; Tτ[j, j]  = 1.0
            end
            continue
        end
        uj = Ucol[j]
        for i in 1:j-1
            if Cnan[i]
                H[i, j] = H[j, i] = NaN
                I[i, j] = I[j, i] = NaN
                R[i, j] = R[j, i] = NaN
                if signed
                    Rsg[i, j] = Rsg[j, i] = NaN
                    Tτ[i, j]  = Tτ[j, i]  = NaN
                end
                continue
            end
            ui = Ucol[i]
            Ub[1, :] .= ui; Ub[2, :] .= uj
            est = η(Ub; k=k, p=p, leafsize=leafsize)
            H[i, j] = H[j, i] = est.H
            I[i, j] = I[j, i] = est.I
            R[i, j] = R[j, i] = est.r

            if signed
                τij = StatsBase.corkendall(hcat(ui, uj))
                Tτ[i, j]  = Tτ[j, i]  = τij[1, 2]
                Rsg[i, j] = Rsg[j, i] = sign(τij[1, 2]) * est.r
            end
        end
    end

    return signed ? (; H, I, r_signed=Rsg) : (; H, I, r=R)
end
function _cortail(X::AbstractMatrix{<:Real}; t = :lower, method = :SchmidtStadtmueller, p = nothing)
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    m, n = size(X)
    n ≥ 2 || throw(ArgumentError("≥ 2 variables (columns) are required."))
    (t === :lower || t === :upper) || throw(ArgumentError("t ∈ {:lower,:upper}"))
    U = t === :upper ? (1 .- Float64.(X)) : Float64.(X)
    anynan = [any(isnan, @view U[:, j]) for j in 1:n]
    p === nothing && (p = 1 / sqrt(m))
    (0 < p < 1) || throw(ArgumentError("p must be in (0,1); hint: p = 1/√m"))

    Lam = Matrix{Float64}(LinearAlgebra.I, n, n)

    if method === :SchmidtStadtmueller
        B = U .<= p
        @inbounds @views for j in 2:n
            anynan[j] && continue
            bj = B[:, j]
            for i in 1:j-1
                if anynan[i]
                    Lam[i,j] = Lam[j,i] = NaN
                else
                    bi = B[:, i]
                    c  = sum(bi .& bj)
                    Lam[i,j] = Lam[j,i] = clamp((c / m) / p, 0.0, 1.0)
                end
            end
        end

    elseif method === :SchmidSchmidt
        pmu = max.(0.0, p .- U)
        S   = Matrix{Float64}(I, n, n)
        @inbounds @views for j in 2:n
            anynan[j] && continue
            y = pmu[:, j]
            for i in 1:j-1
                if anynan[i]
                    S[i,j] = S[j,i] = NaN
                else
                    x = pmu[:, i]
                    S[i,j] = S[j,i] = dot(x, y) / m
                end
            end
        end
        int_over_Pi = (p^2 / 2)^2
        int_over_M  = p^3 / 3
        scale = int_over_M - int_over_Pi
        @inbounds for j in 2:n, i in 1:j-1
            if isfinite(S[i,j])
                Lam[i,j] = Lam[j,i] = clamp((S[i,j] - int_over_Pi) / scale, 0.0, 1.0)
            else
                Lam[i,j] = Lam[j,i] = NaN
            end
        end

    else
        throw(ArgumentError("method must be :SchmidtStadtmueller or :SchmidSchmidt"))
    end
    @inbounds for j in 1:n
        if anynan[j]
            Lam[:, j] .= NaN
            Lam[j, :] .= NaN
            Lam[j, j]  = 1.0
        end
    end
    return Lam
end
corlowertail(X::AbstractMatrix{<:Real}, method = :SchmidtStadtmueller, p=nothing) = _cortail(X; t=:lower, method=method, p=p)
coruppertail(X::AbstractMatrix{<:Real}, method = :SchmidtStadtmueller, p=nothing) = _cortail(X; t=:upper, method=method, p=p)

###############################################################################
#####  Conditioning framework.
#####  User-facing function: `condition(), rosenblatt(), inverse_rosenblatt()`
#####
#####  When implementing new models, you can overwrite: 
#####   - `DistortionFromCop(C::Copula{d}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {d, p}`
#####   - `ConditionalCopula(C::Copula{d}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {d, p}`
###############################################################################


# A fewx utility functions
function _assemble(D, is, js, uᵢₛ, uⱼₛ)
    Tᵢ = eltype(typeof(uᵢₛ)); Tⱼ = eltype(typeof(uⱼₛ)); T = promote_type(Tᵢ, Tⱼ)
    w = fill(one(T), D)
    @inbounds for (k,i) in pairs(is); w[i] = uᵢₛ[k]; end
    @inbounds for (k,j) in pairs(js); w[j] = uⱼₛ[k]; end
    return w
end
function _swap(u, i, uᵢ)
    T = promote_type(eltype(u), typeof(uᵢ))
    v = similar(u, T)
    @inbounds for k in eachindex(u)
        v[k] = u[k]
    end
    v[i] = uᵢ
    return v
end
_der(f, u, i::Int) = ForwardDiff.derivative(uᵢ -> f(_swap(u, i, uᵢ)), u[i])
_der(f, u, is::NTuple{1,Int}) = _der(f, u, is[1])
_der(f, u, is::NTuple{N,Int}) where {N} = _der(u′ -> _der(f, u′, (is[end],)), u, is[1:end-1])
_partial_cdf(C, is, js, uᵢₛ, uⱼₛ) = _der(u -> Distributions.cdf(C, u), _assemble(length(C), is, js, uᵢₛ, uⱼₛ), js)

_process_tuples(::Val{D}, js::NTuple{p, Int64}, ujs::NTuple{p, Float64}) where {D,p} = (js, ujs) 
_process_tuples(::Val{D}, j::Int64, uj::Real) where {D} = ((j,), (uj,)) 
function _process_tuples(::Val{D}, js, ujs) where D
    p, p2 = length(js), length(ujs)
    @assert 0 < p < D "js=$(js) must be a non-empty proper subset of 1:D of length at most D-1 (D = $D)"
    @assert p == p2 && all(0 .<= ujs .<= 1) "uⱼₛ must be in [0,1] and match js length"
    jst = Tuple(collect(Int, js))
    @assert all(in(1:D), jst)
    ujst = Tuple(collect(float.(ujs)))
    return (jst, ujst)
end

"""
    Distortion <: Distributions.ContinuousUnivariateDistribution

Abstract super-type for objects describing the (uniform-scale) conditional marginal
transformation U_i | U_J = u_J of a copula.

Subtypes implement cdf/quantile on [0,1]. They are not full arbitrary distributions;
they model how a uniform variable is distorted by conditioning. They can be applied
as a function to a base marginal distribution to obtain the conditional marginal on
the original scale: if `D::Distortion` and `X::UnivariateDistribution`, then `D(X)`
is the distribution of `X_i | U_J = u_J`.
"""
abstract type Distortion<:Distributions.ContinuousUnivariateDistribution end
(D::Distortion)(::Distributions.Uniform) = D
(D::Distortion)(X::Distributions.UnivariateDistribution) = DistortedDist(D, X)
Distributions.minimum(::Distortion) = 0.0
Distributions.maximum(::Distortion) = 1.0
function Distributions.quantile(d::Distortion, α::Real) 
    T = typeof(float(α))
    ϵ = eps(T)
    α < ϵ && return zero(T)
    α > 1 - 2ϵ && return one(T)
    lα = log(α)
    f(u) = Distributions.logcdf(d, u) - lα
    return Roots.find_zero(f, (ϵ, 1 - 2ϵ), Roots.Bisection(); xtol = sqrt(eps(T)))
end
# You have to implement one of these two: 
Distributions.logcdf(d::Distortion, t::Real) = log(Distributions.cdf(d, t))
Distributions.cdf(d::Distortion, t::Real) = exp(Distributions.logcdf(d, t))

"""
    DistortionFromCop{TC,p,T} <: Distortion

Generic, uniform-scale conditional marginal transformation for a copula.

This is the default fallback (based on mixed partial derivatives computed via
automatic differentiation) used when a faster specialized `Distortion` is not
available for a given copula family.

Parameters
- `TC`: copula type
- `p`: length of the conditioned index set J (static)
- `T`: element type for the conditioned values u_J

Construction
- `DistortionFromCop(C::Copula, js::NTuple{p,Int}, ujs::NTuple{p,<:Real}, i::Int)`
    builds the distortion for the conditional marginal of index `i` given `U_js = ujs`.

Notes
- A convenience method `DistortionFromCop(C, j::Int, uj::Real, i::Int)` exists for
    the common `p = 1` case.
"""
struct DistortionFromCop{TC,p}<:Distortion
    C::TC
    i::Int
    js::NTuple{p,Int}
    uⱼₛ::NTuple{p,Float64}
    den::Float64
    function DistortionFromCop(C::Copula{D}, js, uⱼₛ, i) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        p = length(jst)
        if p==1
            den = Distributions.pdf(subsetdims(C, jst), uⱼₛt[1])
        else
            den = Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        end
        return new{typeof(C), p}(C, i, jst, uⱼₛt, den)
    end
end
Distributions.cdf(d::DistortionFromCop, u::Real) = _partial_cdf(d.C, (d.i,), d.js, (u,), d.uⱼₛ) / d.den

"""
    DistortedDist{Disto,Distrib} <: Distributions.UnivariateDistribution

Push-forward of a base marginal by a `Distortion`.
"""
struct DistortedDist{Disto, Distrib}<:Distributions.ContinuousUnivariateDistribution
    D::Disto
    X::Distrib
    function DistortedDist(D::Distortion, X::Distributions.UnivariateDistribution)
        return new{typeof(D), typeof(X)}(D, X)
    end
end
Distributions.cdf(D::DistortedDist, t::Real) = Distributions.cdf(D.D, Distributions.cdf(D.X, t))
Distributions.quantile(D::DistortedDist, α::Real) = Distributions.quantile(D.X, Distributions.quantile(D.D, α))

"""
    ConditionalCopula{d} <: Copula{d}

Copula of the conditioned random vector U_I | U_J = u_J.
"""
struct ConditionalCopula{d, D, p, TDs}<:Copula{d}
    C::Copula{D}
    js::NTuple{p, Int}
    is::NTuple{d, Int}
    uⱼₛ::NTuple{p, Float64}
    den::Float64
    distortions::TDs
    function ConditionalCopula(C::Copula{D}, js, uⱼₛ) where {D}
        jst, uⱼₛt = _process_tuples(Val{D}(), js, uⱼₛ)
        ist = Tuple(setdiff(1:D, jst))
        p = length(jst)
        d = D - p
        distos = Tuple(DistortionFromCop(C, jst, uⱼₛt, i) for i in ist)
                if p==1
            den = Distributions.pdf(subsetdims(C, jst), uⱼₛt[1])
        else
            den = Distributions.pdf(subsetdims(C, jst), collect(uⱼₛt))
        end
        return new{d, D, p, typeof(distos)}(C, jst, ist, uⱼₛt, den, distos)
    end
end
function _cdf(CC::ConditionalCopula{d,D,p,T}, v::AbstractVector{<:Real}) where {d,D,p,T}
    return _partial_cdf(CC.C, CC.is, CC.js, Distributions.quantile.(CC.distortions, v), CC.uⱼₛ) / CC.den
end

###########################################################################
#####  condition() function
###########################################################################
"""
        condition(C::Copula{D}, js, u_js)
        condition(X::SklarDist, js, x_js)

Construct conditional distributions with respect to a copula, either on the
uniform scale (when passing a `Copula`) or on the original data scale (when
passing a `SklarDist`).

Arguments
- `C::Copula{D}`: D-variate copula
- `X::SklarDist`: joint distribution with copula `X.C` and marginals `X.m`
- `js`: indices of conditioned coordinates (tuple, NTuple, or vector)
- `u_js`: values in [0,1] for `U_js` (when conditioning a copula)
- `x_js`: values on original scale for `X_js` (when conditioning a SklarDist)
- `j, u_j, x_j`: 1D convenience overloads for the common p = 1 case

Returns
- If the number of remaining coordinates `d = D - length(js)` is 1:
    - `condition(C, js, u_js)` returns a `Distortion` on [0,1] describing
        `U_i | U_js = u_js`.
    - `condition(X, js, x_js)` returns an unconditional univariate distribution
        for `X_i | X_js = x_js`, computed as the push-forward `D(X.m[i])` where
        `D = condition(C, js, u_js)` and `u_js = cdf.(X.m[js], x_js)`.
- If `d > 1`:
    - `condition(C, js, u_js)` returns the conditional joint distribution on
        the uniform scale as a `SklarDist(ConditionalCopula, distortions)`.
    - `condition(X, js, x_js)` returns the conditional joint distribution on the
        original scale as a `SklarDist` with copula `ConditionalCopula(C, js, u_js)` and
        appropriately distorted marginals `D_k(X.m[i_k])`.

Notes
- For best performance, pass `js` and `u_js` as NTuple to keep `p = length(js)`
    known at compile time. The specialized method `condition(::Copula{2}, j, u_j)`
    exploits this for the common `D = 2, d = 1` case.
- Specializations are provided for many copula families (Independent, Gaussian, t,
    Archimedean, several bivariate families). Others fall back to an automatic
    differentiation based construction.
- This function returns the conditional joint distribution `H_{I|J}(· | u_J)`.
    The “conditional copula” is `ConditionalCopula(C, js, u_js)`, i.e., the copula
    of that conditional distribution.
"""
condition(C::Copula{D}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
function condition(C::Copula{D}, js::NTuple{p, Int}, uⱼₛ::NTuple{p, Float64}) where {D, p}
    margins = Tuple(DistortionFromCop(C, js, uⱼₛ, i) for i in setdiff(1:D, js))
    p==D-1 && return margins[1]
    return SklarDist(ConditionalCopula(C, js, uⱼₛ), margins)
end

condition(C::SklarDist{<:Copula{D}}, j, xⱼ) where D = condition(C, _process_tuples(Val{D}(), j, xⱼ)...)
function condition(X::SklarDist{<:Copula{D}, Tpl}, js::NTuple{p, Int}, xⱼₛ::NTuple{p, Float64}) where {D, Tpl, p}
    uⱼₛ = Tuple(Distributions.cdf(X.m[j], xⱼ) for (j,xⱼ) in zip(js, xⱼₛ))
    margins = Tuple(DistortionFromCop(X.C, js, uⱼₛ, i)(X.m[i]) for i in setdiff(1:D, js))
    p==D-1 && return margins[1]
    return SklarDist(ConditionalCopula(X.C, js, uⱼₛ), margins)
end

###########################################################################
#####  Methods for conditioning subsetcopulas. 
###########################################################################


function DistortionFromCop(S::SubsetCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p}
    ibase = S.dims[i]
    jsbase = ntuple(k -> S.dims[js[k]], p)
    return DistortionFromCop(S.C, jsbase, uⱼₛ, ibase)
end

function ConditionalCopula(S::SubsetCopula{d,CT}, js, uⱼₛ) where {d,CT}
    Jbase = Tuple(S.dims[j] for j in js)
    CC_base = ConditionalCopula(S.C, Jbase, uⱼₛ)
    D = length(S.C); I = Tuple(setdiff(1:D, Jbase))
    dims_remain = Tuple(i for i in S.dims if !(i in Jbase))
    posmap = Dict(i => p for (p,i) in enumerate(I))
    dims_positions = Tuple(posmap[i] for i in dims_remain)
    return (length(dims_positions) == length(I)) ? CC_base : SubsetCopula(CC_base, dims_positions)
end


###########################################################################
#####  Generic Rosenblatt and inverse Rosenblatt via conditioning
###########################################################################
"""
    rosenblatt(C::Copula, u)

Computes the rosenblatt transform associated to the copula C on the vector u. Formally, assuming that U ∼ C, the result should be uniformely distributed on the unit hypercube. The importance of this transofrmation comes from its bijectivity: `inverse_rosenblatt(C, rand(d))` is equivalent to `rand(C)`. The interface proposes faster versions for matrix inputs `u`.

Generic Rosenblatt transform using conditional distortions:
S₁ = U₁, S_k = H_{k|1:(k-1)}(U_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.


* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
"""
rosenblatt(C::Copula{d}, u::AbstractVector{<:Real}) where {d} = rosenblatt(C, reshape(u, (d, 1)))[:]
function rosenblatt(C::Copula{d}, u::AbstractMatrix{<:Real}) where {d}
    size(u, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    v = similar(u)
    @inbounds for j in axes(u, 2)
        # First coordinate is unchanged
        v[1, j] = clamp(float(u[1, j]), 0.0, 1.0)
        for k in 2:d
            js = ntuple(i -> i, k - 1)
            ujs = ntuple(i -> float(u[i, j]), k - 1)  # condition on original u's
            Dk = DistortionFromCop(C, js, ujs, k)
            v[k, j] = Distributions.cdf(Dk, clamp(float(u[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    v = similar(u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.cdf.(Mᵢ, u[i,:])
    end
    return rosenblatt(D.C, v)
end

"""
    inverse_rosenblatt(C::Copula, u)

Computes the inverse rosenblatt transform associated to the copula C on the vector u. Formally, assuming that U ∼ Π, the independence copula, the result should be distributed as C. Also look at `rosenblatt(C, u)` for the inverse transformation. The interface proposes faster versions for matrix inputs `u`. 

Generic inverse Rosenblatt using conditional distortions:
U₁ = S₁, U_k = H_{k|1:(k-1)}^{-1}(S_k | U₁:U_{k-1}).
Specialized families may provide faster overrides.


References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
"""
inverse_rosenblatt(C::Copula{d}, u::AbstractVector{<:Real}) where {d} = inverse_rosenblatt(C, reshape(u, (d, 1)))[:]
function inverse_rosenblatt(C::Copula{d}, s::AbstractMatrix{<:Real}) where {d}
    size(s, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    v = similar(s)
    @inbounds for j in axes(s, 2)
        v[1, j] = clamp(float(s[1, j]), 0.0, 1.0)
        for k in 2:d
            js = ntuple(i -> i, k - 1)
            ujs = ntuple(i -> float(v[i, j]), k - 1)  # use already reconstructed U's
            Dk = DistortionFromCop(C, js, ujs, k)
            v[k, j] = Distributions.quantile(Dk, clamp(float(s[k, j]), 0.0, 1.0))
        end
    end
    return v
end
function inverse_rosenblatt(D::SklarDist, u::AbstractMatrix{<:Real})
    v = inverse_rosenblatt(D.C,u)
    for (i,Mᵢ) in enumerate(D.m)
        v[i,:] .= Distributions.quantile.(Mᵢ, v[i,:])
    end
    return v
end

###############################################################################
#####  Fitting interface
#####  User-facing function:
#####   - `Distributions.fit(CopulaModel, MyCopulaType, data, method)`
#####   - `Distributions.fit(MyCopulaType, data, method)`
#####
#####  If you want your copula to be fittable byt he default interface, you can overwrite: 
#####   - _available_fitting_methods() to tell the system which method you allow. 
#####   - _fit(MyCopula, data, Val{:mymethod}) to make the fit.  
#####  
#####  Or, for simple models, to get access to a few default bindings, you could also override the following: 
#####   - Distributions.params() yielding a NamedTuple of parameters
#####   - _unbound_params() mappin your parameters to unbounded space
#####   - _rebound_params() doing the reverse
#####   - _example() giving example copula of your type. 
#####   - _example() giving example copula of your type. 
#####  
###############################################################################



"""
    CopulaModel{CT, TM, TD} <: StatsBase.StatisticalModel

A fitted copula model.

This type stores the result of fitting a copula (or a Sklar distribution) to
pseudo-observations or raw data, together with auxiliary information useful
for statistical inference and model comparison.

# Fields
- `result::CT`          — the fitted copula (or `SklarDist`).
- `n::Int`              — number of observations used in the fit.
- `ll::Float64`         — log-likelihood at the optimum.
- `method::Symbol`      — fitting method used (e.g. `:mle`, `:itau`, `:deheuvels`).
- `vcov::Union{Nothing, AbstractMatrix}` — estimated covariance of the parameters, if available.
- `converged::Bool`     — whether the optimizer reported convergence.
- `iterations::Int`     — number of iterations used in optimization.
- `elapsed_sec::Float64` — time spent in fitting.
- `method_details::NamedTuple` — additional method-specific metadata (grid size, pseudo-values, etc.).

`CopulaModel` implements the standard `StatsBase.StatisticalModel` interface:
[`StatsBase.nobs`](@ref), [`StatsBase.coef`](@ref), [`StatsBase.coefnames`](@ref), [`StatsBase.vcov`](@ref),
[`StatsBase.aic`](@ref), [`StatsBase.bic`](@ref), [`StatsBase.deviance`](@ref), etc.

See also [`Distributions.fit`](@ref) and [`_copula_of`](@ref).
"""
struct CopulaModel{CT, TM<:Union{Nothing,AbstractMatrix}, TD<:NamedTuple} <: StatsBase.StatisticalModel
    result        :: CT
    n             :: Int
    ll            :: Float64
    method        :: Symbol
    vcov          :: TM
    converged     :: Bool
    iterations    :: Int
    elapsed_sec   :: Float64
    method_details:: TD
    function CopulaModel(c::CT, n::Integer, ll::Real, method::Symbol;
                         vcov=nothing, converged=true, iterations=0, elapsed_sec=NaN,
                         method_details=NamedTuple()) where {CT}
        return new{CT, typeof(vcov), typeof(method_details)}(
            c, n, float(ll), method, vcov, converged, iterations, float(elapsed_sec), method_details
        )
    end
end

# Fallbacks that throw if the interface s not implemented correctly. 
"""
    Distributions.params(C::Copula)
    Distributions.params(S::SklarDist)

Return the parameters of the given distribution `C`. Our extension gives these parameters in a named tuple format. 

# Arguments
- `C::Distributions.Distribution`: The distribution object whose parameters are to be retrieved. Copulas.jl implements particular bindings for SklarDist and Copula objects. 

# Returns
- A named tuple containing the parameters of the distribution in the order they are defined for that distribution type.
"""
Distributions.params(C::Copula) = throw("You need to specify the Distributions.params() function as returning a named tuple with parameters.")
_example(CT::Type{<:Copula}, d) = throw("You need to specify the `_example(CT::Type{T}, d)` function for your copula type, returning an example of the copula type in dimension d.")
_unbound_params(CT::Type{Copula}, d, θ) = throw("You need to specify the _unbound_param method, that takes the namedtuple returned by `Distributions.params(CT(d, θ))` and trasform it into a raw vector living in R^p.")
_rebound_params(CT::Type{Copula}, d, α) = throw("You need to specify the _rebound_param method, that takes the output of _unbound_params and reconstruct the namedtuple that `Distributions.params(C)` would have returned.")
function _fit(CT::Type{<:Copula}, U, ::Val{:mle})
    # @info "Running the MLE routine from the generic implementation"
    d   = size(U,1)
    function cop(α)
        par = _rebound_params(CT, d, α)
        return CT(d, par...) ####### Using a "," here forces the constructor to accept raw values, while a ";" passes named values. Not sure which is best. 
    end
    α₀  = _unbound_params(CT, d, Distributions.params(_example(CT, d)))

    loss(C) = -Distributions.loglikelihood(C, U)
    res = try
        Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS(); autodiff=:forward)
    catch err
        # @warn "LBFGS with AD failed ($err), retrying with NelderMead"
        Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    end
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return CT(d, θhat...),
           (; θ̂=θhat, optimizer = Optim.summary(res), converged = Optim.converged(res), iterations = Optim.iterations(res))
end

"""
    _fit(::Type{<:Copula}, U, ::Val{method}; kwargs...)

Internal entry point for fitting routines.

Each copula family implements `_fit` methods specialized on `Val{method}`.
They must return a pair `(copula, meta)` where:
- `copula` is the fitted copula instance,
- `meta::NamedTuple` holds method–specific metadata to be stored in `method_details`.

This is not intended for direct use by end–users.  
Use [`Distributions.fit(CopulaModel, ...)`] instead.
"""
function _fit(CT::Type{<:Copula}, U, method::Union{Val{:itau}, Val{:irho}, Val{:ibeta}})
    # @info "Running the itau/irho/ibeta routine from the generic implementation"
    d   = size(U,1)

    cop(α) = CT(d, _rebound_params(CT, d, α)...)
    α₀  = _unbound_params(CT, d, Distributions.params(_example(CT, d)))
    @assert length(α₀) <= d*(d-1)/2 "Cannot use $method since there are too much parameters."

    fun = method isa Val{:itau} ? StatsBase.corkendall : method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est = fun(U')
    loss(C) = sum(abs2, est .- fun(C))

    res = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return CT(d, θhat...),
           (; θ̂=θhat, optimizer = Optim.summary(res), converged = Optim.converged(res), iterations = Optim.iterations(res))
end
"""
    Distributions.fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Quick fit: devuelve solo la cópula ajustada (atajo de `Distributions.fit(CopulaModel, CT, U; summaries=false, kwargs...).result`).
"""
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; summaries=false, kwargs...).result

"""
    _available_fitting_methods(::Type{<:Copula})

Return the tuple of fitting methods available for a given copula family.

This is used internally by [`Distributions.fit`](@ref) to check validity of the `method` argument
and to select a default method when `method=:default`.

# Example
```julia
_available_fitting_methods(GumbelCopula)
# → (:mle, :itau, :irho, :ibeta)
```
"""
_available_fitting_methods(::Type{<:Copula}) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(C::Copula) = _available_fitting_methods(typeof(C))

function _find_method(CT, method)
    avail = _available_fitting_methods(CT)
    isempty(avail) && error("No fitting methods available for $CT.")
    if method === :default 
        method = avail[1]
        # @info "Choosing default method '$(method)' among $avail..."
    elseif method ∉ avail 
        error("Method '$method' not available for $CT. Available: $(join(avail, ", ")).")
    end
    return method
end
"""
    fit(CopulaModel, CT::Type{<:Copula}, U; method=:default, summaries=true, kwargs...)

Fit a copula of type `CT` to pseudo-observations `U`.

# Arguments
- `U::AbstractMatrix` — a `d×n` matrix of data (each column is an observation).
  If the input is raw data, use `SklarDist` fitting instead to estimate both
  margins and copula simultaneously.
- `method::Symbol`    — fitting method; defaults to the first available one
  (see [`_available_fitting_methods`](@ref)).
- `summaries::Bool`   — whether to compute pairwise summary statistics
  (Kendall's τ, Spearman's ρ, Blomqvist's β).
- `kwargs...`         — additional method-specific keyword arguments
  (e.g. `pseudo_values=true`, `grid=401` for extreme-value tails, etc.).

# Returns
A [`CopulaModel`](@ref) containing the fitted copula and metadata.

# Examples
```julia
U = rand(GumbelCopula(2, 3.0), 500)

M = fit(CopulaModel, GumbelCopula, U; method=:mle)
println(M)

# Quick fit: returns only the copula
C = fit(GumbelCopula, U; method=:itau)
```
"""
function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U; method = :default, summaries=true, kwargs...)
    d, n = size(U)
    # Choose the fitting method: 
    method = _find_method(CT, method)

    t = @elapsed (rez = _fit(CT, U, Val{method}(); kwargs...))
    C, meta = rez
    ll = Distributions.loglikelihood(C, U)
    md = (; d, n, method, meta..., null_ll=0.0, elapsed_sec=t, _extra_pairwise_stats(U, !summaries)...)
    return CopulaModel(C, n, ll, method;
        vcov         = get(md, :vcov, nothing),
        converged    = get(md, :converged, true),
        iterations   = get(md, :iterations, 0),
        elapsed_sec  = get(md, :elapsed_sec, NaN),
        method_details = md)
end

_available_fitting_methods(::Type{SklarDist}) = (:ifm, :ecdf)
"""
    fit(CopulaModel, SklarDist{CT, TplMargins}, X; copula_method=:default, sklar_method=:default,
                                           summaries=true, margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple())

Joint margin and copula adjustment (Sklar approach).
`sklar_method ∈ (:ifm, :ecdf)` controls whether parametric CDFs (`:ifm`) or pseudo-observations (`:ecdf`) are used.
"""
function Distributions.fit(::Type{CopulaModel},::Type{SklarDist{CT,TplMargins}}, X; copula_method = :default, sklar_method = :default,
                           summaries = true, margins_kwargs = NamedTuple(), copula_kwargs = NamedTuple()) where
                           {CT<:Copulas.Copula, TplMargins<:Tuple}

    sklar_method = _find_method(SklarDist, sklar_method)
    copula_method = _find_method(CT, copula_method)

    d, n = size(X)
    marg_types = TplMargins.parameters
    (length(marg_types) == d) || throw(ArgumentError("SklarDist: #marginals $(length(marg_types)) ≠ d=$d"))
    m = ntuple(i -> Distributions.fit(marg_types[i], @view X[i, :]; margins_kwargs...), d)
    # Only one margins_kwargs while people mught want to pass diferent kwargs for diferent marginals... but OK for the moment.

    
    U = similar(X)
    if sklar_method === :ifm
        for i in 1:d
            U[i,:] .= Distributions.cdf.(m[i], X[i,:])
        end
    elseif sklar_method === :ecdf
        U .= pseudos(X)
    end

    # Copula fit... with method specific
    C, cmeta = _fit(CT, U, Val{copula_method}(); copula_kwargs...)

    S  = SklarDist(C, m)
    ll = Distributions.loglikelihood(S, X)

    null_ll = 0.0
    @inbounds for j in axes(X, 2)
        for i in 1:d
            null_ll += Distributions.logpdf.(m[i], X[i, j])
        end
    end

    return CopulaModel(S, n, ll, copula_method;
        vcov           = get(cmeta, :vcov, nothing),   # vcov of the copula (if you compute it)
        converged      = get(cmeta, :converged, true),
        iterations     = get(cmeta, :iterations, 0),
        elapsed_sec    = get(cmeta, :elapsed_sec, NaN),
        method_details = (; cmeta..., null_ll, sklar_method, margins = map(typeof, m), 
              has_summaries = summaries, d=d, n=n, _extra_pairwise_stats(U, !summaries)...))
end

function _uppertriangle_stats(mat)
    # compute the mean and std of the upper triangular part of the matrix (diagonal excluded)
    gen = [mat[idx] for idx in CartesianIndices(mat) if idx[1] < idx[2]]
    return Statistics.mean(gen), length(gen) == 1 ? zero(gen[1]) : Statistics.std(gen), minimum(gen), maximum(gen)
end
function _extra_pairwise_stats(U::AbstractMatrix, bypass::Bool)
    bypass && return (;)
    τm, τs, τmin, τmax = _uppertriangle_stats(StatsBase.corkendall(U'))
    ρm, ρs, ρmin, ρmax = _uppertriangle_stats(StatsBase.corspearman(U'))
    βm, βs, βmin, βmax = _uppertriangle_stats(corblomqvist(U'))
    γm, γs, γmin, γmax = _uppertriangle_stats(corgini(U'))
    return (; tau_mean=τm, tau_sd=τs, tau_min=τmin, tau_max=τmax,
             rho_mean=ρm, rho_sd=ρs, rho_min=ρmin, rho_max=ρmax,
             beta_mean=βm, beta_sd=βs, beta_min=βmin, beta_max=βmax,
             gamma_mean=γm, gamma_sd=γs, gamma_min=γmin, gamma_max=γmax)
end

"""
    nobs(M::CopulaModel) -> Int

Number of observations used in the model fit.
"""
StatsBase.nobs(M::CopulaModel)     = M.n
StatsBase.isfitted(::CopulaModel)  = true

"""
    deviance(M::CopulaModel) -> Float64

Deviation of the fitted model (-2 * loglikelihood).
"""
StatsBase.deviance(M::CopulaModel) = -2 * M.ll
StatsBase.dof(M::CopulaModel) = StatsBase.dof(M.result)

"""
    _copula_of(M::CopulaModel)

Returns the copula object contained in the model, even if the result is a `SklarDist`.
"""
_copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

"""
    coef(M::CopulaModel) -> Vector{Float64}

Vector with the estimated parameters of the copula.
"""
StatsBase.coef(M::CopulaModel) = collect(values(Distributions.params(_copula_of(M)))) # why ? params of the marginals should also be taken into account. 

"""
    coefnames(M::CopulaModel) -> Vector{String}

Names of the estimated copula parameters.
"""
StatsBase.coefnames(M::CopulaModel) = string.(keys(Distributions.params(_copula_of(M))))
StatsBase.dof(C::Copulas.Copula) = length(values(Distributions.params(C)))

#(optional vcov) and vcov its very important... for inference 
"""
    vcov(M::CopulaModel) -> Union{Nothing, Matrix{Float64}}

Variance and covariance matrix of the estimators.
Can be `nothing` if not available.
"""
StatsBase.vcov(M::CopulaModel) = M.vcov
function StatsBase.stderror(M::CopulaModel)
    V = StatsBase.vcov(M)
    V === nothing && throw(ArgumentError("stderror: vcov(M) == nothing."))
    return sqrt.(LinearAlgebra.diag(V))
end
function StatsBase.confint(M::CopulaModel; level::Real=0.95)
    V = StatsBase.vcov(M)
    V === nothing && throw(ArgumentError("confint: vcov(M) == nothing."))
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level)/2)
    θ = StatsBase.coef(M)
    se = sqrt.(LinearAlgebra.diag(V))
    return θ .- z .* se, θ .+ z .* se
end

"""
    aic(M::CopulaModel) -> Float64

Akaike information criterion for the fitted model.
"""
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*M.ll

"""
    bic(M::CopulaModel) -> Float64

Bayesian information criterion for the fitted model.
"""
StatsBase.bic(M::CopulaModel) = StatsBase.dof(M)*log(StatsBase.nobs(M)) - 2*M.ll
function aicc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    corr = (n > k + 1) ? (2k*(k+1)) / (n - k - 1) : Inf
    return StatsBase.aic(M) + corr
end
function hqc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    return -2*M.ll + 2k*log(log(max(n, 3)))
end

function StatsBase.nullloglikelihood(M::CopulaModel)
    if hasproperty(M.method_details, :null_ll)
        return getfield(M.method_details, :null_ll)
    else
        throw(ArgumentError("nullloglikelihood not available in method_details."))
    end
end
StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)