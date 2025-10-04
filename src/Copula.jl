###############################################################################
#####  Main Copula interface. 
#####  User-facing function: 
#####       1) Distributions.jl's API: cdf, pdf, logpdf, loglikelyhood, etc..
#####       2) ρ, τ, β, γ, ι, λₗ and λᵤ: repectively the spearman rho, kendall tau,  blomqvist's beta, 
#####          gini's gamma,entropy eta, and lower and upper tail dependencies. 
#####       3) measure(C, us, vs) that get the measure associated with the copula.  
#####       3) pseudo(data) construct pseudo-data from a given dataset.  
#####
#####  When implementing a new copula, you have to overwrite `Copulas._cdf()`
#####  and you may overwrite ρ, τ, β, γ, ι, λₗ, λᵤ, measure for performances. 
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
function γ(C::Copula{d}) where {d}
    _integrand(u) = (1 + minimum(u) - maximum(u) + max(abs(sum(u) - d/2) - (d - 2)/2, 0.0)) / 2
    I = Distributions.expectation(_integrand, C; nsamples=10^4)
    a = 1/(d+1) + 1/factorial(d+1)   # independence
    b = (2 + 4.0^(1-d)) / 3          # comonotonicity
    return (I - a) / (b - a)
end
function ι(C::Copula{d}) where {d}
    return Distributions.expectation(u -> -Distributions.logpdf(C, u), C; nsamples=10^4)
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
    d, n = size(U)
    I = zero(eltype(U))
    for j in 1:n
        u = U[:,j]
        I += (1 + minimum(u) - maximum(u) + max(abs(sum(u) - d/2) - (d - 2)/2, 0.0)) / 2
    end
    I /= n
    a = 1/(d+1) + 1/factorial(d+1)
    b = (2 + 4.0^(1-d)) / 3
    return (I - a) / (b - a)
end
function _λ(U::AbstractMatrix; t::Symbol=:upper, p::Union{Nothing,Real}=nothing)
    # Assumes pseudo-data given. Multivariate tail’s lambda (Schmidt, R. & Stadtmüller, U. 2006)
    p === nothing && (p = 1/sqrt(size(U, 2)))
    (0 < p < 1) || throw(ArgumentError("p must be in (0,1)"))
    in_tail = t=== :upper ? Base.Fix2(>=, 1-p) : Base.Fix2(<=, p)
    prob = Statistics.mean(all(in_tail, U, dims=1))
    return clamp(prob/p, 0.0, 1.0)
end
λₗ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:lower, p=p)
λᵤ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:upper, p=p)
function ι(U::AbstractMatrix; k::Int=5, p::Real=Inf, leafsize::Int=32)
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
    return H
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