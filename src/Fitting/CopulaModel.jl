# ===================== src/Fitting/CopulaModel.jl =====================
struct CopulaModel{C, TM<:Union{Nothing,AbstractMatrix}, TD<:NamedTuple} <: StatsBase.StatisticalModel
    copula        :: C
    n             :: Int
    ll            :: Float64
    method        :: Symbol
    vcov          :: TM
    converged     :: Bool
    iterations    :: Int
    elapsed_sec   :: Float64
    method_details:: TD

    function CopulaModel(c::C, n::Integer, ll::Real, method::Symbol;
                         vcov=nothing, converged=true, iterations=0, elapsed_sec=NaN,
                         method_details=NamedTuple()) where {C}
        return new{C, typeof(vcov), typeof(method_details)}(
            c, n, float(ll), method, vcov, converged, iterations, float(elapsed_sec), method_details
        )
    end
end

copula(M::CopulaModel) = M.copula

# === table for show... ===
function _print_coeftable(io::IO, M::CopulaModel; level::Real=0.95)
    if StatsBase.dof(M) == 0
        return _print_empirical_summary(io, M)  # ver §2
    end

    θ  = StatsBase.coef(M)
    nm = StatsBase.coefnames(M)
    V  = StatsBase.vcov(M)

    if V === nothing || isempty(θ)
        println(io, "────────────────────────────────────────")
        @printf(io, "%-14s %12s\n", "Parameter", "Estimate")
        println(io, "────────────────────────────────────────")
        @inbounds for (j, name) in pairs(nm)
            @printf(io, "%-14s %12.6g\n", String(name), θ[j])
        end
        println(io, "────────────────────────────────────────")
        return
    end

    se = sqrt.(LinearAlgebra.diag(V))
    z  = θ ./ se
    p  = 2 .* Distributions.ccdf(Distributions.Normal(), abs.(z))
    lo, hi = StatsBase.confint(M; level=level)
    lvl = Int(round(100*level))

    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n",
            "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","$lvl% Lo","$lvl% Hi")
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @inbounds for j in eachindex(θ)
        @printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n",
                String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
    end
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
end

function Base.show(io::IO, M::CopulaModel)
    println(io, "$(typeof(copula(M))) fitted via $(M.method)")
    n  = StatsBase.nobs(M)
    ll = Distributions.loglikelihood(M)
    @printf(io, "Number of observations: %d\n", n)

    ll0 = StatsBase.nullloglikelihood(M)
    if isfinite(ll0); @printf(io, "Null Loglikelihood: %.4f\n", ll0); end
    @printf(io, "Loglikelihood: %.4f\n", ll)

    k = StatsBase.dof(M)
    if isfinite(ll0) && k > 0
        LR = 2*(ll - ll0)
        χ² = Distributions.Chisq(k)
        p  = Distributions.ccdf(χ², LR)
        @printf(io, "LR Test: %.2f ∼ χ²(%d) ⟹  Pr > χ² = %.4g\n", LR, k, p)
    end

    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    @printf(io, "AIC: %.3f   BIC: %.3f\n", aic, bic)

    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        it   = M.iterations
        tsec = isfinite(M.elapsed_sec) ? @sprintf("%.3fs", M.elapsed_sec) : "NA"
        println(io, "Converged: $(conv)   Iterations: $(it)   Elapsed: $(tsec)")
    end

    _print_coeftable(io, M)
end

_default_method(::Type{<:ArchimedeanCopula})  = :mle
# ====================== Dispatch ======================
@inline _select_method(::Type{T}, method::Symbol) where {T<:Copula} =
    (method === :auto) ? _default_method(T) : method

@inline function _dispatch_fit(::Type{T}, U::AbstractMatrix, m::Symbol; kwargs...) where {T<:Copula}
    method = (m === :auto) ? _default_method(T) : m

    if T <: ArchimedeanCopula
        if method === :mle
            return _fit_mle_a(T, U; kwargs...)
        elseif method === :itau
            return _fit_itau(T, U; kwargs...)
        elseif method === :irho
            return _fit_irho(T, U; kwargs...)
        elseif method === :ibeta
            return _fit_ibeta(T, U; kwargs...)
        elseif method === :emp
            return _fit_empirical(T, U; kwargs...)  # :beta (default), :bernstein, :raw, EV if T <: EV
        else
            throw(ArgumentError("method=$(method) is not supported for $(T). Use one of: :mle, :itau, :irho, :ibeta, :emp."))
        end

    else
        if method === :emp
            return _fit_empirical(T, U; kwargs...)
        else
            throw(ArgumentError("method=$(method) is not implemented for $(T). "
                                * "Use method=:emp or set a specific Archimedean family."))
        end
    end
end

# --------------------------- Distributions.fit -------------------------------

function Distributions.fit(::Type{CopulaModel}, ::Type{T}, U::AbstractMatrix; method::Symbol=:auto, kwargs...) where {T<:Copula}
    m        = _select_method(T, method)
    C, meta  = _dispatch_fit(T, U, m; kwargs...)
    Up       = _as_pxn(C, U)
    ll       = Distributions.loglikelihood(C, Up)
    n        = size(Up, 2)

    d  = Copulas.length(C)
    ll0 = Distributions.loglikelihood(Copulas.IndependentCopula(d), Up)

    meta = merge(meta, (; null_ll = ll0))

    return CopulaModel(C, n, ll, m;
        vcov       = get(meta, :vcov, nothing),
        converged  = get(meta, :converged, true),
        iterations = get(meta, :iterations, 0),
        elapsed_sec= get(meta, :elapsed_sec, NaN),
        method_details = meta)
end



# ====================== Empirical ======================

# helpers: stast pairwise
@inline function _pairwise_stats(Up::AbstractMatrix)
    d, n = size(Up)
    @assert d ≥ 2
    τ = StatsBase.corkendall(Up')   # d×d, variables en columnas ⇒ Up'
    ρ = StatsBase.corspearman(Up')  # d×d

    npairs = d*(d-1) ÷ 2
    valsτ = Vector{Float64}(undef, npairs)
    valsρ = Vector{Float64}(undef, npairs)
    idx = 1
    @inbounds for i in 1:d-1, j in i+1:d
        valsτ[idx] = τ[i, j]
        valsρ[idx] = ρ[i, j]
        idx += 1
    end

    τμ = StatsBase.mean(valsτ)
    ρμ = StatsBase.mean(valsρ)
    τσ = (npairs > 1 ? StatsBase.std(valsτ) : 0.0)
    ρσ = (npairs > 1 ? StatsBase.std(valsρ) : 0.0)

    return (; tau_mean=τμ, tau_sd=τσ, tau_min=minimum(valsτ), tau_max=maximum(valsτ),
             rho_mean=ρμ, rho_sd=ρσ, rho_min=minimum(valsρ), rho_max=maximum(valsρ))
end

function _fit_empirical(::Type{T}, U::AbstractMatrix;
                        emp_kind::Symbol = :beta,
                        pseudo_values::Bool = true,
                        m::Union{Nothing,Int,Tuple{Vararg{Int}}} = nothing,
                        estimator::Symbol = :ols, # only whe we implement ExtremeValueFit
                        grid::Int = 401, eps::Real = 1e-3,
                        kwargs...) where {T<:Copulas.Copula}
    # data
    d  = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    Up = _as_pxn(d, U)

    # case EV
    if T <: Copulas.ExtremeValueCopula || T === Copulas.EmpiricalEVCopula
        C = EmpiricalEVCopula(Up; estimator=estimator, grid=grid, eps=eps, pseudos=pseudo_values)  # important.... pseudos
        meta = (; estimator=:emp, emp_kind=:ev_tail, pseudo_values, grid, eps, d=d, n=size(Up,2))
        return C, meta
    end

    # No-EV
    if emp_kind === :beta
        C = BetaCopula(Up)
        stats = _pairwise_stats(Up)
        meta = merge((; estimator=:emp, emp_kind=:beta, pseudo_values, d=d, n=size(Up,2),
                       beta_hat=blomqvist_beta(Up)), stats)
        return C, meta

    elseif emp_kind === :bernstein
        base = EmpiricalCopula(Up; pseudo_values=pseudo_values)
        C    = BernsteinCopula(base; m=m)
        stats = _pairwise_stats(Up)
        meta = merge((; estimator=:emp, emp_kind=:bernstein, pseudo_values, m=C.m, d=d, n=size(Up,2),
                       beta_hat=blomqvist_beta(Up)), stats)
        return C, meta

    elseif emp_kind === :raw || emp_kind === :emp
        C = EmpiricalCopula(Up; pseudo_values=pseudo_values)
        stats = _pairwise_stats(Up)
        meta = merge((; estimator=:emp, emp_kind=:raw, pseudo_values, d=d, n=size(Up,2),
                       beta_hat=blomqvist_beta(Up)), stats)
        return C, meta

    else
        throw(ArgumentError("emp_kind ∈ (:beta, :bernstein, :raw); got $(emp_kind)."))
    end
end

# show table of summary when dof==0 (empírico)
function _print_empirical_summary(io::IO, M::CopulaModel)
    md   = M.method_details
    kind = get(md, :emp_kind, :raw)
    d    = get(md, :d, missing)
    n    = get(md, :n, missing)
    pv   = get(md, :pseudo_values, missing)

    # header
    println(io, "Empirical summary ($kind)")
    hdr = "d=$(d), n=$(n)" * (pv === missing ? "" : ", pseudo_values=$(pv)")
    if kind === :bernstein
        m = get(md, :m, nothing); if m !== nothing; hdr *= ", m=$(m)"; end
    elseif kind === :ev_tail
        est = get(md, :estimator, missing)
        grid= get(md, :grid, missing)
        eps = get(md, :eps,  missing)
        hdr *= ", estimator=$(est), grid=$(grid), eps=$(eps)"
    end
    println(io, hdr)

    has_tau  = all(haskey.(Ref(md), (:tau_mean, :tau_sd, :tau_min, :tau_max)))
    has_rho  = all(haskey.(Ref(md), (:rho_mean, :rho_sd, :rho_min, :rho_max)))
    has_beta = haskey(md, :beta_hat)

    if d === missing || d == 2
        println(io, "────────────────────────────")
        @printf(io, "%-10s %12s\n", "Stat", "Value")
        println(io, "────────────────────────────")
        if has_tau;  @printf(io, "%-10s %12.3f\n", "tau",  md[:tau_mean]); end
        if has_rho;  @printf(io, "%-10s %12.3f\n", "rho",  md[:rho_mean]); end
        if has_beta; @printf(io, "%-10s %12.3f\n", "beta", md[:beta_hat]); end
        println(io, "────────────────────────────")
    else
        # Multivariado: mean/sd/min/max por pares
        println(io, "───────────────────────────────────────────────────────")
        @printf(io, "%-10s %10s %10s %10s %10s\n", "Stat","Mean","SD","Min","Max")
        println(io, "───────────────────────────────────────────────────────")
        if has_tau
            @printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "tau", md[:tau_mean], md[:tau_sd], md[:tau_min], md[:tau_max])
        end
        if has_rho
            @printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "rho", md[:rho_mean], md[:rho_sd], md[:rho_min], md[:rho_max])
        end
        if has_beta
            @printf(io, "%-10s %10.3f %10s %10s %10s\n", "beta", md[:beta_hat], "—", "—", "—")
        end
        println(io, "───────────────────────────────────────────────────────")
    end
end
