
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
