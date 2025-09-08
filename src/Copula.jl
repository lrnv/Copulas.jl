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
    r = HCubature.hcubature(F,z,i,rtol=sqrt(eps()))[1]
    return 12*r-3
end
function τ(C::Copula)
    F(x) = Distributions.cdf(C,x)
    r = Distributions.expectation(F,C; nsamples=10^4)
    return 4*r-1
end
function StatsBase.corkendall(C::Copula{d}) where d
    # returns the matrix of bivariate kendall taus.
    K = ones(d,d)
    for i in 1:d
        for j in i+1:d
            K[i,j] = τ(SubsetCopula(C::Copula{d},(i,j)))
            K[j,i] = K[i,j]
        end
    end
    return K
end
function StatsBase.corspearman(C::Copula{d}) where d
    # returns the matrix of bivariate spearman rhos.
    K = ones(d,d)
    for i in 1:d
        for j in i+1:d
            K[i,j] = ρ(SubsetCopula(C::Copula{d},(i,j)))
            K[j,i] = K[i,j]
        end
    end
    return K
end
function measure(C::Copula{d}, u,v) where {d}

    # Computes the value of the cdf at each corner of the hypercube [u,v]
    # To obtain the C-volume of the box.
    # This assumes u[i] < v[i] for all i
    # Based on Computing the {{Volume}} of {\emph{n}} -{{Dimensional Copulas}}, Cherubini & Romagnoli 2009

    # We use a gray code according to the proposal at https://discourse.julialang.org/t/looping-through-binary-numbers/90597/6

    T = promote_type(eltype(u), eltype(v), Float64)
    u .= T.(clamp.(u, 0, 1))
    v .= T.(clamp.(v, 0, 1))
    any(v .≤ u) && return T(0)
    all(iszero.(u)) && all(isone.(v)) && return T(1)

    eval_pt = copy(u)
    # Inclusion–exclusion: the sign for the corner at u is (-1)^d
    # (for d even it's +1, for d odd it's -1). The Gray-code loop below
    # then applies alternating signs matching (-1)^(d - |ε|) as bits flip.
    sign0 = isodd(d) ? -one(T) : one(T)
    r = sign0 * Distributions.cdf(C, eval_pt)
    graycode = 0    # use a gray code to flip one element at a time
    which = fill(false, d) # false/true to use u/v for each component (so false here)
    for s = 1:(1<<d)-1
        graycode′ = s ⊻ (s >> 1)
        graycomp = trailing_zeros(graycode ⊻ graycode′) + 1
        graycode = graycode′
        eval_pt[graycomp] = (which[graycomp] = !which[graycomp]) ? v[graycomp] : u[graycomp]
        # The number of high corners selected equals the number of set bits in the Gray code.
        # Sign = (-1)^(d - k) where k = number of v's in the corner.
        k = count_ones(graycode)
        sign = isodd(d - k) ? -one(T) : one(T)
        r += sign * Distributions.cdf(C, eval_pt)
    end
    return max(r,0)
end