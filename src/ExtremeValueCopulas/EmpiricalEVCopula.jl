# ===================== EmpiricalEVCopula (bivariada) =====================

"""
    EmpiricalEVCopula

Cópula de valores extremos **no paramétrica** en dimensión 2, definida por
una estimación no paramétrica de la función de Pickands `Â(t)` sobre una malla
`tgrid ⊂ (0,1)` y evaluada por interpolación lineal.

# Campos
- `tgrid::Vector{Float64}`  : malla en (0,1).
- `Ahat::Vector{Float64}`   : Pickands estimada y proyectada (convexa, `max(t,1-t)≤A≤1`).

# Constructores

    EmpiricalEVCopula(u; estimator=:ols, grid=401, eps=1e-3, pseudos=true)

- `u`: matriz de datos de tamaño (2,N) o (N,2).
- `estimator ∈ {:ols,:cfg,:pickands}` (por defecto `:ols`).
- `grid`: puntos en la malla de `t`.
- `eps`: recorte de bordes para `tgrid=(eps,1-eps)`.
- `pseudos`: si `false`, se convierten `u` a pseudo-observaciones con `pseudos(u)`.

**Nota:** La proyección a la clase de Pickands (convexidad+cotas) se aplica
siempre por estabilidad y para garantizar que se obtiene una EV–cópula válida.

# Referencias
- Pickands (1981), Capéraà–Fougères–Genest (1997), OLS (intercepto) con
  representación simplificada y eficiencia asintótica (ver la literatura estándar).
"""
struct EmpiricalEVCopula <: ExtremeValueCopula{Float64}
    tgrid::Vector{Float64}
    Ahat::Vector{Float64}
end

Base.eltype(::EmpiricalEVCopula) = Float64


# EmpiricalEVCopula.jl

# Interpolación lineal compatible con ForwardDiff.Dual
@inline function _A_lin_interp(tgrid::Vector{Float64}, Â::Vector{Float64}, t)
    tv = ForwardDiff.value(t)  # Float64: sólo para búsqueda de índice
    # bordes → devolver tipo(t)
    if tv <= 0.0
        return one(t)
    elseif tv >= 1.0
        return one(t)
    end

    idx = searchsortedlast(tgrid, tv)
    if idx <= 0
        return Â[1] + zero(t)           # promueve a Dual si hace falta
    elseif idx >= length(tgrid)
        return Â[end] + zero(t)
    else
        tL = tgrid[idx]
        tR = tgrid[idx+1]
        AL = Â[idx]
        AR = Â[idx+1]

        # peso en Float64 pero promovido al tipo de t
        w  = (tv - tL) / (tR - tL)
        wd = w + zero(t)

        ALd = AL + zero(t)
        ARd = AR + zero(t)

        return (one(t) - wd)*ALd + wd*ARd
    end
end

# ---------- Implementación de A para la EV no paramétrica ----------

# A(C,t) es requerida por tu jerarquía EV; se evalúa por interpolación
function A(C::EmpiricalEVCopula, t::Real)
    _A_lin_interp(C.tgrid, C.Ahat, t)
end

function EmpiricalEVCopula(u::AbstractMatrix;
                           estimator::Symbol=:ols, grid::Int=401, eps::Real=1e-3,
                           pseudos::Bool=true)

    Up = pseudos ? u : pseudos(u)
    d  = size(Up,1) ≥ 2 ? size(Up,1) : size(Up,2)
    d == 2 || throw(ArgumentError("EmpiricalEVCopula: sólo implementado para d=2."))

    tgrid = collect(range(eps, 1-eps; length=grid))

    Â = estimator === :ols      ? empirical_pickands_ols(tgrid, Up; endpoint_correction=true) :
         estimator === :cfg      ? empirical_pickands_cfg(tgrid, Up; endpoint_correction=true) :
         estimator === :pickands ? empirical_pickands(tgrid, Up; endpoint_correction=true) :
         throw(ArgumentError("estimator ∈ {:ols,:cfg,:pickands}"))

    # Proyección SIEMPRE: garantiza EV–cópula válida y deriva numéricamente estable
    _convexify_pickands!(Â, tgrid)

    return EmpiricalEVCopula(tgrid, Â)
end
