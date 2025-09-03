# =========================== src/Fitting/stats.jl ===========================

# Normalized to p×n (rows = var, cols = samples)
@inline _as_pxn(C::Copula, U::AbstractMatrix) =
    (size(U,1) == length(C)) ? U : permutedims(U)

@inline _as_pxn(p::Integer, U::AbstractMatrix) =
    (size(U,1) == p) ? U : permutedims(U)

function Distributions.loglikelihood(C::Copula, U::AbstractMatrix{<:Real})
    Up = _as_pxn(C, U)
    total = 0.0
    @inbounds @views @simd for j in axes(Up, 2)
        total += Distributions.logpdf(C, Up[:, j])
    end
    return total
end

Distributions.loglikelihood(C::Copula, u::AbstractVector{<:Real}) =
    Distributions.logpdf(C, u)

# ================== Empirical tail dependence (d = 2) ==================
@inline function _ranks1(x::AbstractVector)
    p = sortperm(x)                    # O(n log n)
    r = similar(p)
    @inbounds for i in 1:length(p)
        r[p[i]] = i
    end
    return r
end
@inline function _as2xn(U::AbstractMatrix)
    (size(U,1) == 2) ? U : permutedims(U)
end
@inline function _ranks2(U::AbstractMatrix)
    U2 = _as2xn(U)
    return _ranks1(view(U2,1,:)), _ranks1(view(U2,2,:))
end
@inline function _k_window(n::Integer; c1::Float64=0.6, c2::Float64=1.8, kmin_abs::Int=10)
    s  = sqrt(float(n))
    k1 = max(kmin_abs, floor(Int, c1*s))
    k2 = max(k1+5,    floor(Int, c2*s))
    k2 = min(k2, n-1)
    return k1:k2
end

# Kernel of Schmidt–Stadtmüller/Huang for a k 
@inline function _lambda_upper_ss_with_ranks(R1::AbstractVector{<:Integer},
                                             R2::AbstractVector{<:Integer},
                                             k::Int)
    n = length(R1)
    thr = n - k
    cnt = 0
    @inbounds @simd for j in 1:n
        cnt += (R1[j] > thr && R2[j] > thr) ? 1 : 0
    end
    return cnt / k
end

@inline function _lambda_lower_ss_with_ranks(R1::AbstractVector{<:Integer},
                                             R2::AbstractVector{<:Integer},
                                             k::Int)
    n = length(R1)
    cnt = 0
    @inbounds @simd for j in 1:n
        cnt += (R1[j] ≤ k && R2[j] ≤ k) ? 1 : 0
    end
    return cnt / k
end
# --- util: median without dependences (use partialsort!) ---
@inline function _median!(a::Vector{Float64})
    n = length(a)
    if isodd(n)
        k = (n + 1) >>> 1             # (n+1)/2
        return partialsort!(a, k)
    else
        k = n >>> 1                    # n/2
        x = partialsort!(a, k)
        y = partialsort!(a, k + 1)
        return 0.5*(x + y)
    end
end
# χ²: Yi = Σ (Φ^{-1}(U'j))^2
function _score_chi(up::AbstractVector)
    s = 0.0
    @inbounds for v in up
        vv = clamp(v, 1e-12, 1 - 1e-12)         # skip ±Inf in quantile
        z  = Distributions.quantile(_N01, vv)
        s += z*z
    end
    return s
end

# Γ: Yi = -Σ log(U'j)
function _score_gamma(up::AbstractVector)
    s = 0.0
    @inbounds for v in up
        vv = clamp(v, 1e-12, 1 - 1e-12)
        s += -log(vv)
    end
    return s
end

# --- Univariate distances ---
function _dist_cvm!(Y::Vector{Float64}, F::Function)
    sort!(Y)
    n  = length(Y)
    nf = float(n)
    s  = 1.0/(12.0*nf)
    @inbounds for (i, y) in enumerate(Y)
        u  = (2.0*i - 1.0)/(2.0*nf)
        Fy = clamp(F(y), 0.0, 1.0)
        d  = u - Fy
        s += d*d
    end
    return s
end

function _dist_ks!(Y::Vector{Float64}, F::Function)
    sort!(Y)
    n  = length(Y)
    nf = float(n)
    mx = 0.0
    @inbounds for (i, y) in enumerate(Y)
        Fn_left  = (i-1)/nf
        Fn_right = i/nf
        Fy = clamp(F(y), 0.0, 1.0)
        mx = max(mx, Fy - Fn_left, Fn_right - Fy)
    end
    return mx
end
# Distancia CvM en [0,1] sobre malla tgrid:
# Aemp, Ath, w :: Vector{Float64},  Δ :: Float64 (paso de la malla)
@inline function _dist_cvm(Aemp::AbstractVector{<:Real},
                           Ath::AbstractVector{<:Real},
                           w::AbstractVector{<:Real},
                           Δ::Real)
    @assert length(Aemp) == length(Ath) == length(w)
    s = 0.0
    @inbounds @simd for i in eachindex(Aemp)
        d = float(Aemp[i]) - float(Ath[i])
        s += float(w[i]) * d*d
    end
    return s * float(Δ)
end

# Distancia KS: sup_i |Aemp(t_i) - Ath(t_i)|
@inline function _dist_ks(Aemp::AbstractVector{<:Real},
                          Ath::AbstractVector{<:Real})
    @assert length(Aemp) == length(Ath)
    m = 0.0
    @inbounds @simd for i in eachindex(Aemp)
        d = abs(float(Aemp[i]) - float(Ath[i]))
        if d > m; m = d; end
    end
    return m
end
# ============================ API pública ============================

function upper_tail(U::AbstractMatrix)
    R1, R2 = _ranks2(U)
    n = length(R1)
    ks = _k_window(n)
    vals = Vector{Float64}(undef, length(ks))
    @inbounds for (i,k) in pairs(ks)
        vals[i] = _lambda_upper_ss_with_ranks(R1, R2, k)
    end
    m = _median!(vals)
    return clamp(m, 0.0, 1.0)
end

function lower_tail(U::AbstractMatrix)
    R1, R2 = _ranks2(U)
    n = length(R1)
    ks = _k_window(n)
    vals = Vector{Float64}(undef, length(ks))
    @inbounds for (i,k) in pairs(ks)
        vals[i] = _lambda_lower_ss_with_ranks(R1, R2, k)
    end
    m = _median!(vals)
    return clamp(m, 0.0, 1.0)
end

@inline function _winsorize_tau_vclib(τ::Float64)
    s = τ < 0 ? -1.0 : 1.0
    a = abs(τ)
    a = a < 0.01 ? 0.01 : (a > 0.9 ? 0.9 : a)
    return s*a
end

# β̂ multivariate (Hofert–Mächler–McNeil, ec. (7))
function blomqvist_beta(U::AbstractMatrix)
    d, n = size(U)
    c = 2.0^(d-1) / (2.0^(d-1) - 1.0)
    acc = 0.0
    @inbounds for i in 1:n
        q1 = all(@view(U[:, i]) .<= 0.5)
        q3 = all(@view(U[:, i]) .>  0.5)
        acc += (q1 || q3) ? 1.0 : 0.0
    end
    return c * (acc/n - 2.0^(1-d))
end

_open(lo, hi) = (isfinite(lo) ? lo+_δθ : -5.0,
                 isfinite(hi) ? hi-_δθ :  5.0)

function _bracket_beta!(f, target, lo, hi)
    a, b = _open(lo, hi)
    fa = f(a) - target
    fb = f(b) - target
    if isfinite(lo) && isfinite(hi)
        return a, b, fa, fb
    end
    if !isfinite(hi)
        for t in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
            b = t
            fb = f(b) - target
            sign(fa) != sign(fb) && return a, b, fa, fb
        end
    end
    if !isfinite(lo)
        for t in (-1.0, -2.0, -5.0, -10.0, -20.0, -50.0, -100.0)
            a = t
            fa = f(a) - target
            sign(fa) != sign(fb) && return a, b, fa, fb
        end
    end
    return a, b, fa, fb
end

####### Empirical functions for mde
# ====================== Empíricos de Pickands (2D) ======================

"""
    empirical_pickands(tgrid, U; endpoint_correction=true)

Estimador clásico de Pickands:
Âᴾ(t) = 1 / mean( ξ_i(t) ),   ξ_i(t) = min((-log U_i)/(1-t), (-log V_i)/t).
"""
function empirical_pickands(tgrid::AbstractVector, U::AbstractMatrix;
                            endpoint_correction::Bool=true)
    Up = _as_pxn(U)
    lu = @views -log.(Up[1, :])
    lv = @views -log.(Up[2, :])

    Â = similar(tgrid, Float64)
    @inbounds for (k, t) in pairs(tgrid)
        tt = clamp(t, eps(), 1 - eps())
        ξ  = min.(lu ./ (1 - tt), lv ./ tt)
        Â[k] = 1.0 / StatsBase.mean(ξ)
    end
    if endpoint_correction
        Â[firstindex(Â)] = 1.0
        Â[lastindex(Â)]  = 1.0
    end
    return Â
end

"""
    empirical_pickands_cfg(tgrid, U; endpoint_correction=true)

Estimador CFG (Capéraà–Fougères–Genest):
Âᴄᴴᴳ(t) = exp( -γ - mean(log ξ_i(t)) ).
"""
function empirical_pickands_cfg(tgrid::AbstractVector, U::AbstractMatrix;
                                endpoint_correction::Bool=true)
    Up = _as_pxn(U)
    lu = @views -log.(Up[1, :])
    lv = @views -log.(Up[2, :])

    Â = similar(tgrid, Float64)
    @inbounds for (k, t) in pairs(tgrid)
        tt = clamp(t, eps(), 1 - eps())
        ξ  = min.(lu ./ (1 - tt), lv ./ tt)
        Â[k] = exp(-_EULER_GAMMA - StatsBase.mean(log.(ξ)))
    end
    if endpoint_correction
        Â[firstindex(Â)] = 1.0
        Â[lastindex(Â)]  = 1.0
    end
    return Â
end

"""
    empirical_pickands_ols(tgrid, U; endpoint_correction=true)

Estimador OLS (intercepto):
para cada t, ajusta y_i(t) = -log ξ_i(t) - γ sobre X_i = (1, -log(-log U_i)-γ, -log(-log V_i)-γ);
Âᴼᴸˢ(t) = exp(β̂₀(t)).
"""
function empirical_pickands_ols(tgrid::AbstractVector, U::AbstractMatrix;
                                endpoint_correction::Bool=true)
    Up = _as_pxn(U)
    n  = size(Up, 2)

    lu = @views -log.(Up[1, :])                # -log U_i
    lv = @views -log.(Up[2, :])                # -log V_i
    x1 = @views -log.(lu) .- _EULER_GAMMA      # -log ξ_i(e1) - γ
    x2 = @views -log.(lv) .- _EULER_GAMMA      # -log ξ_i(e2) - γ

    # Diseño con intercepto Z = [1 x1 x2] y proyección izquierda P = (Z'Z)^(-1)Z'
    Z = Matrix{Float64}(undef, n, 3)
    @inbounds begin
        Z[:,1] .= 1.0
        Z[:,2] .= x1
        Z[:,3] .= x2
    end
    ZtZ = LinearAlgebra.Symmetric(Z'Z)
    F   = LinearAlgebra.cholesky(ZtZ)             # SPD en práctica
    P   = F \ (Z')

    Â = similar(tgrid, Float64)
    @inbounds for (k, t) in pairs(tgrid)
        tt = clamp(t, eps(), 1 - eps())
        ξt = min.(lu ./ (1 - tt), lv ./ tt)
        y  = @. -log(ξt) - _EULER_GAMMA
        β  = P * y
        Â[k] = exp(β[1])           # intercepto
    end
    if endpoint_correction
        Â[firstindex(Â)] = 1.0
        Â[lastindex(Â)]  = 1.0
    end
    return Â
end

# ================== Proyección a la clase de Pickands ==================
# Impone:  max(t,1-t) ≤ Â(t) ≤ 1  y convexidad por PAV ponderado.
function _convexify_pickands!(Â::Vector{Float64}, t::Vector{Float64})
    n = length(Â); @assert n == length(t)
    # 1) Cotas
    @inbounds for i in 1:n
        Â[i] = clamp(Â[i], max(t[i], 1 - t[i]), 1.0)
    end
    # 2) Convexidad (PAV con pesos Δt)
    Δt = diff(t)                                # n-1
    s  = [(Â[i+1]-Â[i])/Δt[i] for i=1:n-1]    # pendientes por tramo
    W  = copy(Δt)                               
    C  = ones(Int, n-1)                         # cuántos tramos originales agrupa cada bloque

    i = 1
    while i < length(s)
        if s[i] <= s[i+1] + 1e-14
            i += 1
        else
            newW = W[i] + W[i+1]
            newS = (s[i]*W[i] + s[i+1]*W[i+1]) / newW
            s[i] = newS; W[i] = newW; C[i] += C[i+1]
            deleteat!(s, i+1); deleteat!(W, i+1); deleteat!(C, i+1)
            if i > 1; i -= 1; end
        end
    end

    # Expansión a n-1 pendientes
    s_exp = similar(Δt)
    pos = 1
    @inbounds for j in 1:length(s)
        cnt = C[j]
        for _ in 1:cnt
            s_exp[pos] = s[j]
            pos += 1
        end
    end
    @assert pos-1 == length(Δt)

    # Reconstrucción y clamp final
    Â[2:end] = Â[1] .+ cumsum(s_exp .* Δt)
    @inbounds for i in 1:n
        Â[i] = clamp(Â[i], max(t[i], 1 - t[i]), 1.0)
    end
    return Â
end