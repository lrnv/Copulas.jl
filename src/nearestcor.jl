# Turn a matrix of pairwise correlation-like values into a valid correlation
# matrix: symmetrize, set the diagonal to one, and when the result is not
# positive definite, apply the alternating projects method. A final adjustment
# may be applied to ensure that there are no small (machine-precision) negative
# eigenvalues.
# Used both to start the elliptical likelihood optimizers and to repair the
# pairwise rank-inversion estimates in dimension d > 2, where a matrix of
# pairwise closed-form inversions need not be positive definite.
function _nearest_correlation(R0::AbstractMatrix)
    n = size(R0, 1)
    T = float(eltype(R0))
    X = Matrix{T}(R0)

    # precheck
    if any(x -> !isfinite(x), X)
        return Matrix{T}(LinearAlgebra.I, n, n)
    end

    # prepare the matrix
    X .= Matrix(LinearAlgebra.Symmetric((X + X') / 2))
    @inbounds for j in 1:n
        X[j, j] = one(T)
    end

    # matrix already PD?
    δ = sqrt(eps(T))
    λmin = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(X))
    λmin > δ && return X

    m = 2 # history size for acceleration
    tau = sqrt(eps(T)) # minimum enforced eigenvalue
    maxiters = clamp(n, 20, 200)
    reltol = sqrt(eps(T))

    # Initialize working matrices
    Y = copy(X)
    S = zeros(T, n, n)
    R = similar(X)
    G = similar(X)
    Z = similar(X) # scratch space matrix

    # Pre-allocate memory for Anderson Acceleration history
    vec_dim = n * n
    DF = Matrix{T}(undef, vec_dim, m)
    DG = Matrix{T}(undef, vec_dim, m)
    f_old = Vector{T}(undef, vec_dim)
    g_old = Vector{T}(undef, vec_dim)
    m_eff = 0

    # loop statistics
    iter = 0
    resid = Inf

    # projects `src` onto the set of positive semi-definite matrices
    function project_psd!(dst, src, δ, scratch)
        λ, P = LinearAlgebra.eigen(LinearAlgebra.Symmetric(src))
        for i in eachindex(λ)
            λ[i] = max(λ[i], δ)
        end
        Λ = LinearAlgebra.Diagonal(λ)
        Xs = parent(dst)
        LinearAlgebra.mul!(scratch, P, Λ)   # P * Λ  -> Z
        LinearAlgebra.mul!(Xs, scratch, P') # Z * P' -> X
        return dst
    end

    while iter < maxiters
        iter += 1

        R .= Y .- S
        project_psd!(X, R, tau, Z)
        S .= X .- R
        copyto!(G, X)
        @inbounds for j in 1:n
            G[j, j] = one(T)
        end

        resid = LinearAlgebra.norm(X .- G) / LinearAlgebra.norm(X)

        if resid <= reltol
            break
        end

        # Anderson Acceleration update step
        # Residual F = G - Y
        f = vec(G .- Y)
        g = vec(G)

        if iter > 1
            df = f .- f_old
            dg = g .- g_old

            # Update memory history buffers without dynamic re-allocations
            if m_eff < m
                m_eff += 1
                DF[:, m_eff] .= df
                DG[:, m_eff] .= dg
            else
                # Shift left
                DF[:, 1:(m - 1)] .= @view DF[:, 2:m]
                DF[:, m] .= df
                DG[:, 1:(m - 1)] .= @view DG[:, 2:m]
                DG[:, m] .= dg
            end

            # Solve linear least-squares: min || DF * gamma - f ||_2
            DF_view = @view DF[:, 1:m_eff]
            DG_view = @view DG[:, 1:m_eff]

            gamma = DF_view \ f

            # Compute accelerated iterate g_acc = g - DG * gamma
            g_acc = g .- DG_view * gamma

            # Reshape back to n x n matrix
            Y .= reshape(g_acc, n, n)

            # Enforce symmetry and unit diagonal on Y for numerical stability
            Y .= (Y .+ Y') ./ 2
            @inbounds for j in 1:n
                Y[j, j] = one(T)
            end
        else
            Y .= G
        end

        copyto!(f_old, f)
        copyto!(g_old, g)
    end

    # ensure unit diagonal
    @inbounds for j in 1:n
        X[j, j] = one(T)
    end

    # final step to ensure PD
    k = 0
    while k < 5
        LinearAlgebra.isposdef(X) && break
        project_psd!(X, X, δ, Z)
        StatsBase.cov2cor!(X)
        δ = sqrt(δ) # increase δ towards 1
        k += 1
    end

    return X
end
