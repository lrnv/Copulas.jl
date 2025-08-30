# ============================ Constante numérica ============================
const _EPS_LOG = 1e-12

# ============================ NLL (1 parámetro, genérico) ===========================

@inline function _ev_nll_1p(::Type{CT}, U::AbstractMatrix, θ::Float64) where {CT<:ExtremeValueCopula}
    C = try
        CT(θ)
    catch
        return Inf
    end
    Up = _as_pxn(U)
    ll = Distributions.loglikelihood(C, Up)
    return isfinite(ll) ? -ll : Inf
end

# ---- Override estable para Cuadras–Augé (masa en diagonal) ----
@inline function _ev_nll_1p(::Type{CuadrasAugeCopula}, U::AbstractMatrix, α::Float64)
    C = try
        CuadrasAugeCopula(α)
    catch
        return Inf
    end
    Up = _as_pxn(U)
    n  = size(Up,2)

    ll = 0.0
    @inbounds for j in 1:n
        u = Up[1,j]; v = Up[2,j]
        # dominio abierto (evita log(0))
        if !(0.0 < u < 1.0 && 0.0 < v < 1.0)
            return Inf
        end
        if abs(log(u) - log(v)) ≤ _EPS_LOG
            # contribución singular en la diagonal
            w = u                     # (= v)
            dens = α * w^(1 - α)
            if !(dens > 0.0) || !isfinite(dens); return Inf; end
            ll += log(dens)
        else
            # parte absolutamente continua
            dens = (1 - α) * (u < v ? v^(-α) : u^(-α))
            if !(dens > 0.0) || !isfinite(dens); return Inf; end
            ll += log(dens)
        end
    end
    return -ll
end

# ============================ Solver 1D (Brent) ============================
function _solve_1d_brent(f::Function, lo::Float64, hi::Float64;
                         maxiter::Int=_MAXITER, xtol::Real=_XTOL,
                         ftol::Real=_FTOL, θ0::Float64=NaN)
    (lo < hi) || throw(ArgumentError("lo ≥ hi in 1D solver."))
    # Optim.Brent no usa semilla; mapeamos tolerancias correctamente
    return Optim.optimize(f, lo, hi, Optim.Brent();
                          abs_tol=xtol, rel_tol=ftol, iterations=maxiter)
end

# ============================ MLE unificado (k = 1,2,3) ============================
function fit_ev_mle(::Type{CT}, U::AbstractMatrix;
                lo_hi = nothing,
                start = :auto,
                maxiter::Int=_MAXITER,
                xtol::Real=_XTOL,
                ftol::Real=_FTOL,
                use_grad::Bool=true) where {CT<:ExtremeValueCopula}

    # 0) Cotas prácticas y dimensión (derivadas de _bounds(CT))
    lo_def, hi_def = param_bounds(CT)
    lo, hi = (lo_hi === nothing) ? (copy(lo_def), copy(hi_def)) : lo_hi
    k = length(lo); @assert k == length(hi)

    if k == 1
        # --- 1D (Brent) ---
        lo1, hi1 = float(lo[1]), float(hi[1])
        θ0 = (start === :auto) ? (isfinite(lo1) && isfinite(hi1) ? (lo1 + 0.381966f0*(hi1 - lo1)) : 1.0) :
             (start isa Real ? float(start) : throw(ArgumentError("invalid start for k=1")))
        lo_eff1, hi_eff1 = lo1, hi1
        if !isfinite(lo1) || !isfinite(hi1)
            (le, he) = _finite_box([lo1], [hi1], [θ0]; width=25.0)
            lo_eff1, hi_eff1 = le[1], he[1]
        end
        (lo_eff1 < hi_eff1) || throw(ArgumentError("Degenerate bounds in ev_mle(k=1)."))

        f = θ -> _ev_nll_1p(CT, U, θ)
        res = _solve_1d_brent(f, lo_eff1, hi_eff1; maxiter=maxiter, xtol=xtol, ftol=ftol, θ0=θ0)
        θ̂ = Optim.minimizer(res)
        if isfinite(lo1) && θ̂ ≤ lo1; θ̂ = nextfloat(lo1); end
        if isfinite(hi1) && θ̂ ≥ hi1; θ̂ = prevfloat(hi1); end
        return CT(θ̂)
    end

    # --- k ∈ {2,3}: caja + LBFGS (fallback Nelder–Mead) ---
    lof = Float64[lo...];  hif = Float64[hi...]
    seed = if start === :auto
        lof .+ 0.1
    elseif start isa AbstractVector
        Float64[start...]
    else
        throw(ArgumentError("start debe ser :auto o Vector para k=$(k)."))
    end
    @assert length(seed) == k

    fθ = if k == 2
        θ -> _nll_two_params(CT, U, θ[1], θ[2])
    elseif k == 3
        θ -> _nll_three_param(CT, U, θ[1], θ[2], θ[3])
    else
        throw(ArgumentError("ev_mle implementado para k∈{1,2,3}."))
    end

    res = _solve_boxed(fθ, lof, hif, seed;
                       maxiter=maxiter, xtol=xtol, ftol=ftol, use_grad=use_grad)
    θ̂ = Optim.minimizer(res)
    return make(CT, θ̂)
end

# ======= Semiparamétric MDE (Pickands) =======
# ====================== MDE semiparamétrico (Pickands) ======================
# ====================== Utilidades distancia/opt. ======================

@inline function _scalar_bounds(lo, hi)
    l = (lo isa AbstractVector || lo isa Tuple) ? lo[1] : lo
    u = (hi isa AbstractVector || hi isa Tuple) ? hi[1] : hi
    return float(l), float(u)
end

# --- solver vectorial en caja (igual al de MLE) ---
function _solve_boxed_vec(f::Function, lo::Vector{Float64}, hi::Vector{Float64}, seed::Vector{Float64};
                          maxiter::Int=1000, xtol::Real=1e-8, ftol::Real=1e-10, use_grad::Bool=false)
    x0 = similar(seed)
    @inbounds for i in eachindex(seed)
        x0[i] = clamp(seed[i], nextfloat(lo[i]), prevfloat(hi[i]))
    end
    if use_grad
        try
            g = θ -> ForwardDiff.gradient(f, θ)
            return Optim.optimize(θ -> f(θ), θ -> g(θ), lo, hi, x0,
                                  Optim.Fminbox(Optim.LBFGS()),
                                  Optim.Options(x_abstol=xtol, f_abstol=ftol,
                                                g_abstol=sqrt(ftol), iterations=maxiter))
        catch
            # cae al método sin gradiente
        end
    end
    return Optim.optimize(f, lo, hi, x0,
                          Optim.Fminbox(Optim.NelderMead()),
                          Optim.Options(x_abstol=xtol, f_abstol=ftol, iterations=maxiter))
end

# --- objetivo MDE en función del número de parámetros ---
# Devuelve un closure θ -> distancia(Â, A(Cθ))
function _mde_obj_pickands(::Type{CT},
                           Ahat::Vector{Float64}, tgrid, w::Vector{Float64}, Δ::Float64,
                           distance::Symbol, k::Int) where {CT<:ExtremeValueCopula}

    dist = (distance === :cvm) ? (Ath -> _dist_cvm(Ahat, Ath, w, Δ)) :
           (distance === :ks  ? (Ath -> _dist_ks(Ahat, Ath)) :
            error("distance ∈ {:cvm,:ks}"))

    if k == 1
        return θ -> begin
            Cθ = CT(θ[1])
            dist(A.(Ref(Cθ), tgrid))
        end
    elseif k == 2
        return θ -> begin
            Cθ = _build_twoparam(CT, θ[1], θ[2])
            dist(A.(Ref(Cθ), tgrid))
        end
    elseif k == 3
        return θ -> begin
            Cθ = _build_threeparam(CT, θ[1], θ[2], θ[3])
            dist(A.(Ref(Cθ), tgrid))
        end
    else
        error("MDE Pickands implementado aquí para k ∈ {1,2,3}.")
    end
end

# --- API principal: el mismo nombre que tu 1-parám, ahora general ---
# --- util: log seguro para A(t) ---
@inline _safe_logA(A::AbstractVector) = log.(clamp.(A, 1e-12, 1.0))

function ev_mde_pickands(::Type{CT}, U::AbstractMatrix;
                                                        estimator::Symbol=:ols,
                                                        distance::Symbol=:cvm,          # :cvm | :ks | :cvm_log | :ks_log
                                                        grid::Int=201, eps::Real=1e-3,
                                                        weight::Symbol=:beta,
                                                        lo_hi=nothing,
                                                        start=:mid,
                                                        maxiter::Int=1000, xtol::Real=1e-8, ftol::Real=1e-10,
                                                        use_grad::Bool=false,
                                                        λ_pen::Real=0.0, α_scale::Real=4.0
                                                    ) where {CT<:ExtremeValueCopula}

    # 1) malla y pesos
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d == 2 || throw(ArgumentError("Sólo implementado para d=2."))
    tgrid = collect(range(eps, 1-eps; length=grid))
    Δ = step(range(eps, 1-eps; length=grid))

    w = similar(tgrid, Float64)
    if weight === :uniform
        fill!(w, 1.0)
    elseif weight === :beta
        ω = 0.25
        @. w = (tgrid^ω) * ((1 - tgrid)^ω)
    else
        throw(ArgumentError("weight ∈ {:uniform,:beta}"))
    end

    # 2) Â(t): estimador empírico elegido
    Â = estimator === :ols      ? empirical_pickands_ols(tgrid, U; endpoint_correction=true) :
         estimator === :cfg      ? empirical_pickands_cfg(tgrid, U; endpoint_correction=true) :
         estimator === :pickands ? empirical_pickands(tgrid, U; endpoint_correction=true) :
         throw(ArgumentError("estimator ∈ {:ols,:cfg,:pickands}"))

    # *** Convexificación y cotas SIEMPRE ***
    _convexify_pickands!(Â, tgrid)

    # 3) cotas y dimensión del parámetro
    lo, hi = lo_hi === nothing ? _bounds(CT) : lo_hi
    length(lo) == length(hi) || error("lo/hi con longitud distinta")
    k = length(lo)
    lo = Float64[lo...]; hi = Float64[hi...]
    # Warning de estabilidad si k>1
    if k == 2
        @warn "MDE para familias de 2 parámetros puede ser inestable. \
            Recomendamos probar MLE o un enfoque no paramétrico."
    elseif k == 3
        @warn "MDE para familias de 3 parámetros es altamente inestable. \
            Considere usar MLE o no paramétrico."
    end
    # 4) semilla
    seed = if start === :mid
        0.5 .* (lo .+ hi)
    elseif start isa AbstractVector
        Float64[start...]
    else
        error("start ∈ {:mid, Vector}")
    end
    length(seed) == k || error("seed de longitud $k requerida")

    # 5) objetivo y optimización
    use_log = (distance === :cvm_log || distance === :ks_log)
    Â_use  = use_log ? _safe_logA(Â) : Â

    fθ = θ -> begin
        Cθ  = (k==1 ? CT(θ[1]) :
               k==2 ? _build_twoparam(CT, θ[1], θ[2]) :
                      _build_threeparam(CT, θ[1], θ[2], θ[3]))
        Ath = A.(Ref(Cθ), tgrid)
        Ath_use = use_log ? _safe_logA(Ath) : Ath

        dist = if distance === :cvm || distance === :cvm_log
            _dist_cvm(Â_use, Ath_use, w, Δ)
        else
            _dist_ks(Â_use, Ath_use)
        end

        pen = (λ_pen > 0 && k ≥ 1) ? λ_pen * (θ[1]/α_scale)^2 : 0.0
        return dist + pen
    end

    res = _solve_boxed_vec(fθ, lo, hi, seed; maxiter=maxiter, xtol=xtol, ftol=ftol, use_grad=use_grad)
    θ̂v = Optim.minimizer(res)

    Ĉ = k==1 ? CT(θ̂v[1]) :
         k==2 ? _build_twoparam(CT, θ̂v[1], θ̂v[2]) :
                 _build_threeparam(CT, θ̂v[1], θ̂v[2], θ̂v[3])

    return Ĉ
end
