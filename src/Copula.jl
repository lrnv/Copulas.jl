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
function γ(C::Copula{d}; nmc::Int=100_000,
                        rng::Random.AbstractRNG=Random.MersenneTwister(123)) where {d}
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
function entropy(C::Copula{d}; nmc::Int=100_000, rng::Random.AbstractRNG=Random.MersenneTwister(123)) where {d}
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
λₗ(C::Copula{d}; ε::Float64 = 1e-10) where {d} = begin
    f(e) = Distributions.cdf(C, fill(e, d)) / e
    clamp(2*f(ε/2) - f(ε), 0.0, 1.0)
end
λᵤ(C::Copula{d}; ε::Float64 = 1e-10) where {d} = begin
    Sc   = SurvivalCopula(C, Tuple(1:d))
    f(e) = Distributions.cdf(Sc, fill(e, d)) / e
    clamp(2*f(ε/2) - f(ε), 0.0, 1.0)
end

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
StatsBase.corkendall(C::Copula{d}) where d = _as_biv(τ, C)
StatsBase.corspearman(C::Copula{d}) where d = _as_biv(ρ, C)
corblomqvist(C::Copula{d}) where d = _as_biv(β, C)
corgini(C::Copula{d}) where d = _as_biv(γ, C)

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