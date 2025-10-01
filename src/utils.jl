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
            est = ι(Ub; k=k, p=p, leafsize=leafsize)
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
