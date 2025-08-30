# ========================= src/Fitting/FitCopula.jl ==========================

_default_method(::Type{<:ArchimedeanCopula})  = :itau
_default_method(::Type{<:EllipticalCopula})   = :itau
_default_method(::Type{<:ExtremeValueCopula}) = :mle
_default_method(::Type{EmpiricalEVCopula})    = :emp
_default_method(::Type{EmpiricalCopula})      = :emp
_default_method(::Type{<:Copula})             = :mle

_default_method(::Type{<:GaussianCopula}) = :itau
_default_method(::Type{<:TCopula})        = :mle
_default_method(::Type{<:BB1Copula}) = :mle
_default_method(::Type{<:BB6Copula}) = :mle
_default_method(::Type{<:BB7Copula}) = :mle
_default_method(::Type{<:BB8Copula}) = :mle
_default_method(::Type{<:BB9Copula}) = :mle
_default_method(::Type{<:BB10Copula}) = :mle

@inline _select_method(::Type{T}, method::Symbol) where {T<:Copula} =
    (method === :auto) ? _default_method(T) : method

# ====================== src/Fitting/Dispatch.jl ======================

@inline function _dispatch_fit(::Type{T}, U::AbstractMatrix, m::Symbol; kwargs...) where {T<:Copula}
    method = (m === :auto) ? _default_method(T) : m

    if method === :mle
        if T <: Union{BB1Copula,BB6Copula,BB7Copula,BB8Copula,BB9Copula,BB10Copula}
            # BB* (tu bloque especial de MLE)
            return fit_bb_mle(T, U; kwargs...)

        elseif T === BC2Copula
            throw(ArgumentError("MLE disabled for BC2Copula for now."))

        elseif T <: GaussianCopula
            # Elíptica gaussiana
            return fit_gaussian_mle(T, U; kwargs...)

        elseif T <: TCopula
            # Elíptica t (ECM ligero)
            return fit_t_mle(T, U; kwargs...)

        elseif T <: ExtremeValueCopula
            # Todas las EV paramétricas (incluye asimétricas)
            return fit_ev_mle(T, U; kwargs...)

        elseif T <: ArchimaxCopula
            # Totalmente paramétrica archimax (si la tienes implementada)
            return fit_arch_mle(T, U; kwargs...)

        else
            # Fallback genérico
            return fit_mle(T, U; kwargs...)
        end

    elseif method === :mde
        if T <: Union{BB1Copula,BB6Copula,BB7Copula,BB8Copula,BB9Copula,BB10Copula}
            return fit_mde_kendall(T, U; kwargs...)

        elseif T <: ExtremeValueCopula && T !== BC2Copula
            return ev_mde_pickands(T, U; kwargs...)

        elseif T <: GaussianCopula
            @warn "MDE no está soportado para GaussianCopula; use :itau/:irho/:ibeta o :mle."
            return fit_gaussian_mle(T, U; kwargs...)

        elseif T <: TCopula
            @warn "MDE no está soportado para TCopula; use :mle (ECM) o un método rank-based si lo implementas."
            return fit_t_mle(T, U; kwargs...)

        else
            return fit_mde(T, U; kwargs...)
        end

    elseif method === :itau
        if T <: GaussianCopula
            return fit_gaussian_itau(T, U)
        elseif T <: TCopula
            @warn "itau para TCopula no fijará ν; usando :mle para completar parámetros."
            return fit_t_mle(T, U; kwargs...)
        else
            return fit_itau(T, U; kwargs...)
        end

    elseif method === :irho
        if T <: GaussianCopula
            return fit_gaussian_irho(T, U)
        elseif T <: TCopula
            @warn "irho para TCopula no fijará ν; usando :mle."
            return fit_t_mle(T, U; kwargs...)
        else
            return fit_irho(T, U; kwargs...)
        end

    elseif method === :ibeta
        if T <: GaussianCopula
            return fit_gaussian_ibeta(T, U)
        elseif T <: TCopula
            @warn "ibeta para TCopula no fijará ν; usando :mle."
            return fit_t_mle(T, U; kwargs...)
        else
            return fit_ibeta(T, U; kwargs...)
        end

    elseif method === :emp
        if T === EmpiricalEVCopula
            # EV no paramétrica directa
            return EmpiricalEVCopula(U; kwargs...)

        elseif T <: ExtremeValueCopula
            @warn "method=:emp ignora la familia $(T) y ajusta una EV no paramétrica (EmpiricalEVCopula)."
            return EmpiricalEVCopula(U; kwargs...)

        elseif T === EmpiricalCopula
            # Empírica multivariada general
            return EmpiricalCopula(U; kwargs...)

        else
            # Para cualquier no-EV, usar la empírica general si lo deseas
            return EmpiricalCopula(U; kwargs...)
        end

    else
        throw(ArgumentError("Method $(method) not supported for $(T)."))
    end
end

# --------------------------- Distributions.fit -------------------------------

function Distributions.fit(::Type{T}, U::AbstractMatrix; method::Symbol=:auto, kwargs...) where {T<:Copula}
    m = _select_method(T, method)
    return _dispatch_fit(T, U, m; kwargs...)
end

function Distributions.fit(::Type{CopulaModel}, ::Type{T}, U::AbstractMatrix; method::Symbol=:auto, kwargs...) where {T<:Copula}
    C  = Distributions.fit(T, U; method=method, kwargs...)
    Up = _as_pxn(U)
    ll = Distributions.loglikelihood(C, Up)
    n  = size(Up, 2)
    m  = _select_method(T, method)
    return CopulaModel(C, n, ll, m)
end

function Distributions.fit(::Type{EmpiricalEVCopula}, u::AbstractMatrix;
                           estimator::Symbol=:ols, grid::Int=401, eps::Real=1e-3,
                           pseudos::Bool=true)
    return EmpiricalEVCopula(u; estimator=estimator, grid=grid, eps=eps, pseudos=pseudos)
end

function Distributions.fit(::Type{EmpiricalEVCopula}, u; pseudos::Bool=true)
    return EmpiricalCopula(u; pseudos=pseudos)
end
