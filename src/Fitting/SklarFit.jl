# ----------------- Core: ajuste IFM (Inference for Margins) -----------------
"""
    fit(SklarDist, CT, X; margins=Normal, method=:auto, clamp_eps=1e-12, kwargs...)

Ajusta un modelo de Sklar con cópula `CT` a los datos `X` (d×n), vía IFM:
1) Ajusta márgenes univariadas con `Distributions.fit`.
2) Transforma a U por PIT.
3) Ajusta la cópula `CT` sobre U con tu dispatcher `fit(CT, U; method=...)`.

- `margins`: `Type` único (para todas) o `NTuple{d,Type}` heterogéneo.
- `method`: método para la **cópula** (`:mle`, `:mde`, `:emp`, …).
- `kwargs...`: pasan al ajuste de la cópula (p.ej. `estimator=:ols` en EV–MDE).
"""
function Distributions.fit(::Type{SklarDist}, ::Type{CT}, X::AbstractMatrix{<:Real};
                           margins=Distributions.Normal, method::Symbol=:auto,
                           clamp_eps::Real=1e-12, kwargs...) where {CT<:Copula}

    d  = (size(X,1) >= 2) ? size(X,1) : size(X,2)
    d ≥ 2 || throw(ArgumentError("SklarDist.fit: se requiere d≥2."))
    Xd = (size(X,1) == d) ? X : permutedims(X)

    # Tipos marginales (homogéneo u heterogéneo)
    MT = (margins isa Tuple || margins isa NTuple) ? margins : ntuple(_ -> margins, d)
    length(MT) == d || error("margins debe tener longitud d = $d.")

    # Ajuste de márgenes con chequeo de dominio para LogNormal
    m = ntuple(i -> begin
        Ti = MT[i]
        if Ti <: Distributions.LogNormal
            Xi = @view Xd[i,:]
            if any(Xi .<= 0)
                throw(ArgumentError("La marginal LogNormal requiere x>0; hay valores ≤ 0 en la variable $i. \
Use otra familia (p.ej. Normal) o transforme los datos."))
            end
        end
        Distributions.fit(Ti, @view Xd[i,:])
    end, d)

    # PIT → U
    U = Array{Float64}(undef, d, size(Xd,2))
    @inbounds for i in 1:d
        @views U[i,:] = Distributions.cdf.(m[i], Xd[i,:])
    end
    @. U = clamp(U, clamp_eps, 1 - clamp_eps)

    # Ajuste de la CÓPULA CT
    C = if CT === EmpiricalEVCopula
        # EmpiricalEVCopula NO acepta `method`; pasar solo sus kwargs propios
        est = get(kwargs, :estimator, :ols)
        grd = get(kwargs, :grid,      401)
        eps = get(kwargs, :eps,       1e-3)
        pse = get(kwargs, :pseudos,   true)
        Distributions.fit(EmpiricalEVCopula, U; estimator=est, grid=grd, eps=eps, pseudos=pse)
    else
        # Resto de copulas: usa tu dispatcher con `method`
        Distributions.fit(CT, U; method=method, kwargs...)
    end

    return SklarDist(C, m)
end

"""
    fit(SklarDist, X; copula=EmpiricalCopula, margins=Normal, method=:auto, clamp_eps=1e-12, kwargs...)

Atajo equivalente a `fit(SklarDist, copula, X; ...)`.
"""
function Distributions.fit(::Type{SklarDist}, X::AbstractMatrix{<:Real};
                           copula::Type{CT}=EmpiricalCopula,
                           margins=Distributions.Normal, method::Symbol=:auto,
                           clamp_eps::Real=1e-12, kwargs...) where {CT<:Copula}

    return Distributions.fit(SklarDist, CT, X;
                             margins=margins, method=method,
                             clamp_eps=clamp_eps, kwargs...)
end

"""
    fit(SklarDist, C::Copula, X; margins=Normal, clamp_eps=1e-12)

Construye `SklarDist` con cópula **ya ajustada** `C`, ajustando solo márgenes sobre `X`.
"""
function Distributions.fit(::Type{SklarDist}, C::Copula, X::AbstractMatrix{<:Real};
                           margins=Distributions.Normal, clamp_eps::Real=1e-12)

    d  = length(C)
    Xd = (size(X,1) == d) ? X : permutedims(X)
    size(Xd,1) == d || throw(ArgumentError("Dimensión de datos inconsistente con la cópula."))

    MT = (margins isa Tuple || margins isa NTuple) ? margins : ntuple(_ -> margins, d)
    length(MT) == d || error("margins debe tener longitud d = $d.")

    m = ntuple(i -> Distributions.fit(MT[i], @view Xd[i,:]), d)
    return SklarDist(C, m)
end
