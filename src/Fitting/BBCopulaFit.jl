# ============================== common utilities ============================

const _XTOL = 1e-6
const _FTOL = 1e-8
const _MAXITER = 800

@inline _as_pxn(U::AbstractMatrix) = (size(U,1) == 2) ? U : permutedims(U)

# Generic NLL for bivariate copulas with 2 parameters (θ, δ)
@inline function _nll_two_param(::Type{T}, U::AbstractMatrix, θ::Float64, δ::Float64) where {T<:Copula}
    C = try
        T(θ, δ)
    catch
        return Inf
    end
    Up = _as_pxn(U)
    ll = Distributions.loglikelihood(C, Up)
    return isfinite(ll) ? -ll : Inf
end

# Boxed optimizer (LBFGS + AD; Nelder–Mead fallback)
function _solve_boxed(f::Function, lo::Vector{Float64}, hi::Vector{Float64}, seed::Vector{Float64};
                      maxiter::Int=_MAXITER, xtol::Real=_XTOL, ftol::Real=_FTOL, use_grad::Bool=true)
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
        end
    end
    return Optim.optimize(f, lo, hi, x0,
                          Optim.Fminbox(Optim.NelderMead()),
                          Optim.Options(x_abstol=xtol, f_abstol=ftol, iterations=maxiter))
end
# ============================= MLE para BB (2p) ==============================

function fit_bb_mle(::Type{CT}, U::AbstractMatrix;
                     maxiter::Int=_MAXITER, xtol::Real=_XTOL, ftol::Real=_FTOL,
                     lo_hi=nothing, start=:auto) where {CT<:Copula}

    # 1) límites prácticos
    lo, hi = lo_hi === nothing ? _bounds(CT) : lo_hi
    npar = length(lo)
    (npar == 2) || throw(ArgumentError("_copula_mle está definido aquí sólo para copulas de 2 parámetros; recibido npar=$(npar) para $(CT)."))

    lof = Float64[lo...]; hif = Float64[hi...]

    # 2) semilla
    seed = if start === :auto
        lof .+ 0.1
    elseif start isa AbstractVector
        s = Float64[start...]
        length(s) == 2 || throw(ArgumentError("start debe tener longitud 2."))
        s
    else
        throw(ArgumentError("start debe ser :auto o Vector{<:Real} de longitud 2."))
    end

    # 3) objetivo y optimización
    fθ = θ -> _nll_two_param(CT, U, θ[1], θ[2])

    res = _solve_boxed(fθ, lof, hif, seed;
                       maxiter=maxiter, xtol=xtol, ftol=ftol, use_grad=true)

    θ̂ = Optim.minimizer(res)
    return _build_twoparam(CT, θ̂[1], θ̂[2])
end