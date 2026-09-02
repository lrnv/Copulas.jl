"""
    ExtremeValueCopula{d, TT}

Constructor

    ExtremeValueCopula(d, tail::Tail)
    ExtremeValueCopula{d}(tail::Tail)

Extreme-value copulas model tail dependence via a stable tail dependence function (STDF) ``\\ell`` or, equivalently,
via a Pickands dependence function ``A``. In any dimension ``d``, the copula cdf is

```math
\\displaystyle C(u) = \\exp\\!\\left(-\\, \\ell(-\\log u_1,\\ldots,-\\log u_d) \\right).
```

For ``d=2``, write ``x=-\\log u``, ``y=-\\log v``, ``s=x+y``, and ``t = x/s``. The relation between ``\\ell`` and ``A`` is

```math
\\ell(x,y) = s\\, A(t), \\qquad A:[0,1]\\to[1/2,1], \\quad A(0)=A(1)=1, \\ A \\text{ convex}.
```

Usage
- Provide any valid tail `tail::Tail` (which implements `A` and/or `ℓ`) to construct the copula.
- Sampling, cdf, and logpdf follow the standard `Distributions.jl` API.

Example
```julia
C = ExtremeValueCopula(2, GalambosTail(θ))
U = rand(C, 1000)
logpdf.(Ref(C), eachcol(U))
```

References:

* [gudendorf2010extreme](@cite) G., & Segers, J. (2010). Extreme-value copulas. In Copula Theory and Its Applications (pp. 127-145). Springer.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
* [mai2014financial](@cite) Mai, J. F., & Scherer, M. (2014). Financial engineering with copulas explained (p. 168). London: Palgrave Macmillan.
"""
struct ExtremeValueCopula{d,TT<:Tail} <: Copula{d}
    tail::TT
    function ExtremeValueCopula{d}(tail::Tail) where {d}
        d >= 2 || throw(ArgumentError("an extreme-value copula requires d ≥ 2"))
        _is_valid_in_dim(tail, d) || throw(ArgumentError(
            "$(typeof(tail)) is not valid in dimension $d",
        ))
        return new{d,typeof(tail)}(tail)
    end
end

function copula_measure_style(C::ExtremeValueCopula{d}) where {d}
    limit_kind(C.tail, Val(d)) === M_LIMIT &&
        return NonAbsolutelyContinuousMeasure()
    return tail_measure_style(C.tail)
end

ExtremeValueCopula(d::Int, tail::Tail) = ExtremeValueCopula{d}(tail)
@inline function _wrap_extreme_value(::Val{d}, tail::TT) where {d,TT<:Tail}
    return invoke(ExtremeValueCopula{d}, Tuple{Tail}, tail)::ExtremeValueCopula{d,TT}
end

@inline _ev_encoded_dimension(CT) = Base.unwrap_unionall(CT).parameters[1]

# Canonical constructor: FamilyCopula{d}(params...).
#
# If `d` is not encoded in the type, non-structured parameterizations must use
# the runtime convenience form FamilyCopula(d, params...). Structured family
# constructors may provide more specific methods that infer `d` from a matrix
# or vector.
function (CT::Type{<:ExtremeValueCopula{d}})(args...; kwargs...) where {d}
    tail = tailof(CT)(args...; kwargs...)
    return _wrap_extreme_value(Val(d), tail)
end

# Resolve the only generic intersection left by integer-valued parameters:
# for FamilyCopula{d}(first::Int, ...), `first` is a parameter; for the
# unparameterized FamilyCopula(first::Int, ...), it is the runtime dimension.
function (CT::Type{<:ExtremeValueCopula{D}})(first::Int, args...; kwargs...) where {D}
    d = _ev_encoded_dimension(CT)
    if d isa TypeVar
        tail = tailof(CT)(args...; kwargs...)
        return _wrap_extreme_value(Val(first), tail)
    end
    tail = tailof(CT)(first, args...; kwargs...)
    return _wrap_extreme_value(Val(d), tail)
end

# Runtime-dimension form for an unparameterized named family alias.
function (CT::Type{<:ExtremeValueCopula})(d::Int, args...; kwargs...)
    tail = tailof(CT)(args...; kwargs...)
    return _wrap_extreme_value(Val(d), tail)
end

@inline function _cdf(C::ExtremeValueCopula{d}, u) where {d}
    kind = limit_kind(C.tail, Val(d))
    kind === M_LIMIT && return minimum(u)
    return _ev_cdf(C, u)
end

function _ev_cdf(C::ExtremeValueCopula{2,<:BivariatePickandsTail}, u)
    u1, u2 = u
    z = zero(u1 + u2)
    o = one(u1 + u2)
    (u1 <= z || u2 <= z) && return z
    u1 >= o && return min(o, u2)
    u2 >= o && return min(o, u1)

    x, y = -log(u1), -log(u2)
    s = x + y
    return exp(-s * A(C.tail, x / s))
end

_ev_cdf(C::ExtremeValueCopula, u) = exp(-ℓ(C.tail, .- log.(u)))
Distributions.params(C::ExtremeValueCopula) = Distributions.params(C.tail)

# Density selection follows Julia dispatch directly. BivariatePickandsTail
# families retain the native scalar Pickands derivative kernel in d=2.
function _bivariate_pickands_logpdf(C, u)
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return -Inf
    (isone(u1) || isone(u2)) && return -Inf
    x, y = -log(u1), -log(u2)
    val, du, dv, dudv = _biv_der_ℓ(C.tail, (x, y))
    core = -dudv + du * dv
    core ≤ 0 && return -Inf
    # Group the exponent contribution so independence (val == x + y,
    # core == 1) returns exactly zero instead of a cancellation residual.
    return log(core) + (x + y - val)
end

_ev_logpdf(C::ExtremeValueCopula{2,<:BivariatePickandsTail}, u) =
    _bivariate_pickands_logpdf(C, u)

function _ev_logcdf_partial(C::ExtremeValueCopula, u, I)
    all(ui -> zero(ui) < ui <= one(ui), u) || return oftype(float(first(u)), -Inf)
    x = -log.(u)
    val = ℓ(C.tail, x)
    logpos = logneg = oftype(val, -Inf)
    partials = Dict{Tuple{Vararg{Int}},Tuple{Int,typeof(val)}}()

    for π in Combinatorics.partitions(collect(I))
        sgn = isodd(length(I) + length(π)) ? -1 : 1
        logabs = zero(val)
        nonzero = true
        for block in π
            block_I = Tuple(block)
            blocksgn, blocklog = get!(partials, block_I) do
                _ellpartial_signlog(C.tail, x, block_I)
            end
            if iszero(blocksgn)
                nonzero = false
                break
            end
            sgn *= blocksgn
            logabs += blocklog
        end
        nonzero || continue
        if sgn > 0
            logpos = LogExpFunctions.logaddexp(logpos, logabs)
        else
            logneg = LogExpFunctions.logaddexp(logneg, logabs)
        end
    end

    isfinite(logpos) || return oftype(val, -Inf)
    if isfinite(logneg)
        logneg < logpos || return oftype(val, -Inf)
        logpos = LogExpFunctions.logsubexp(logpos, logneg)
    end
    return -val - sum(log(u[i]) for i in I) + logpos
end

# Generic d-dimensional density from the mixed STDF partials and the
# partition formula for absolutely continuous extreme-value copulas.
@inline function Distributions._logpdf(C::ExtremeValueCopula{d}, u) where {d}
    kind = limit_kind(C.tail, Val(d))
    if kind === M_LIMIT
        return all(x -> x == first(u), u) ? zero(eltype(u)) : eltype(u)(-Inf)
    end
    return _ev_logpdf(C, u)
end

function _ev_logpdf(C::ExtremeValueCopula{d}, u) where {d}
    any(isone, u) && return oftype(float(first(u)), -Inf)
    return _ev_logcdf_partial(C, u, 1:d)
end

# Conditioning needs mixed CDF partials, not derivatives of the numerical
# implementation used to evaluate the STDF. In particular, multivariate
# Hüsler--Reiss and extremal-t contain Float64 probability kernels that cannot
# accept ForwardDiff dual numbers, while their STDF partials are available
# directly through `_ellpartial_signlog`.
function _partial_cdf(C::ExtremeValueCopula, is, js, uᵢₛ, uⱼₛ)
    u = _assemble(length(C), is, js, uᵢₛ, uⱼₛ)
    logvalue = _ev_logcdf_partial(C, u, js)
    return isfinite(logvalue) ? exp(logvalue) : zero(float(first(u)))
end
function τ(C::ExtremeValueCopula{2})
    limit_kind(C.tail, Val(2)) === M_LIMIT && return 1.0
    return QuadGK.quadgk(
        t -> d²A(C.tail, t) * t * (1 - t) / max(A(C.tail, t), _δ(t)),
        0.0,
        1.0,
    )[1]
end
ρ(C::ExtremeValueCopula{2}) = 12 * QuadGK.quadgk(t -> 1 / (1 + A(C.tail, t))^2, 0.0, 1.0)[1] - 3
β(C::ExtremeValueCopula{2}) = 4^(1 - A(C.tail, 0.5)) - 1
λᵤ(C::ExtremeValueCopula{2}) = 2 * (1 - A(C.tail, 0.5))
λₗ(C::ExtremeValueCopula{2}) =  A(C.tail, 0.5) > 0.5 ? 0.0 : 1.0
function τ⁻¹(::Type{T},τ_val) where {T<:ExtremeValueCopula{2}}
    return τ⁻¹(tailof(T),τ_val)
end


# Sampling is selected directly by tail capability. Families with a preferable
# exact sampler may specialize `_rand!` for their concrete copula type.
function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2,<:BivariatePickandsTail},
    X::AbstractMatrix{T},
) where {T<:Real}
    limit_kind(C.tail, Val(2)) === M_LIMIT && return _rand_M!(rng, X)
    E = ExtremeDist(C.tail)
    for i in axes(X, 2)
        z = rand(rng, E)
        w = rand(rng) < _ghoudi_mixture_probability(C.tail, z) ? rand(rng) : rand(rng) * rand(rng)
        a = A(C.tail, z)
        X[1, i] = exp(log(w) * z / a)
        X[2, i] = exp(log(w) * (1 - z) / a)
    end
    return X
end

function DistortionFromCop(
    C::ExtremeValueCopula{2,TT},
    js::NTuple{1,Int},
    uⱼₛ::NTuple{1,Float64},
    ::Int,
) where {TT}
    limit_kind(C.tail, Val(2)) === M_LIMIT &&
        return MDistortion(float(uⱼₛ[1]), Int8(js[1]))
    return BivEVDistortion(C.tail, Int8(js[1]), float(uⱼₛ[1]))
end

tailof(S::Type{<:ExtremeValueCopula}) = fieldtype(S, :tail)

##############################################################################################################################
####### Fitting functions for univariate tails only (Extreme Value Copulas).
##############################################################################################################################

_example(CT::Type{<:ExtremeValueCopula}, d) =
    ExtremeValueCopula{d}(tailof(CT)(;
        _rebound_params(CT, d, fill(0.01, fieldcount(tailof(CT))))...,
    ))
_fit_copula(::Type{<:ExtremeValueCopula}, d, θ, example) =
    ExtremeValueCopula{d}(tailof(typeof(example))(θ...))
_unbound_params(CT::Type{<:ExtremeValueCopula}, d, θ) = _unbound_params(tailof(CT), d, θ)
_rebound_params(CT::Type{<:ExtremeValueCopula}, d, α) = _rebound_params(tailof(CT), d, α)

_available_fitting_methods(::Type{ExtremeValueCopula}, d) = (:ols, :cfg, :pickands)
_available_fitting_methods(CT::Type{<:ExtremeValueCopula}, d) = (:mle,)
_available_fitting_methods(CT::Type{<:ExtremeValueCopula{2,GT} where {GT<:OneParameterPickandsTail}}, d) =  (:mle, :itau, :irho, :ibeta, :iupper)

# Fitting empírico (OLS, CFG, Pickands):
function _fit(::Type{ExtremeValueCopula}, U, method::Union{Val{:ols}, Val{:cfg}, Val{:pickands}}; pseudo_values=true, grid::Int=401, eps::Real=1e-3, kwargs...)
    m = typeof(method).parameters[1]
    if size(U, 1) == 2
        C = EmpiricalEVCopula(U; method=m, grid=grid, eps=eps, pseudo_values=pseudo_values, kwargs...)
        return C, (; pseudo_values, method=m, grid, eps)
    end
    C = EmpiricalEVCopula(U; method=m, pseudo_values=pseudo_values, kwargs...)
    return C, (; pseudo_values, method=m, degree=C.tail.degree, projection_rmse=C.tail.projection_rmse)
end
function _fit(CT::Type{<:ExtremeValueCopula{d, GT} where {d, GT<:OneParameterPickandsTail}}, U, m::Union{Val{:itau}, Val{:irho}, Val{:ibeta}})
    TT = tailof(typeof(_example(CT, 2)))
    θ = m isa Val{:itau} ? τ⁻¹(CT,  StatsBase.corkendall(U')[1,2]) :
        m isa Val{:irho} ? ρ⁻¹(CT,  StatsBase.corspearman(U')[1,2]) :
                           β⁻¹(CT,  corblomqvist(U')[1,2])
    lo, hi = _θ_bounds(TT, 2)
    # unbounded limits are bound to 1e16 (inf) and zero is bound to (1e-16) for stability
    θ = clamp(θ, iszero(lo) ? 1e-16 : lo, isinf(hi) ? 1e16 : hi)
    return ExtremeValueCopula{2}(TT(θ)), (; θ̂=(θ=θ,))
end
function _fit(CT::Type{<:ExtremeValueCopula{d, GT} where {d, GT<:OneParameterPickandsTail}}, U, ::Val{:iupper})
    TT = tailof(typeof(_example(CT, 2)))
    θ = clamp(λᵤ⁻¹(CT, λᵤ(U)), _θ_bounds(TT, 2)...)
    return ExtremeValueCopula{2}(TT(θ)), (; θ̂=(θ=θ,))
end

function _fit(CT::Type{<:ExtremeValueCopula{d, GT} where {d, GT<:OneParameterPickandsTail}}, U, ::Val{:mle}; start::Union{Symbol,Real}=:itau, xtol::Real=1e-8)
    d = size(U,1)
    example = _example(CT, d)
    ConcreteCT = typeof(example)
    TT = tailof(ConcreteCT)
    lo, hi = _θ_bounds(TT, d)
    θ0_val = if start isa Real
        start
    else
        initial_params = start ∈ (:itau, :irho, :ibeta, :iupper) ? _fit(CT, U, Val{start}())[2].θ̂ : only(Distributions.params(example))
        initial_params.θ
    end
    # Keep the starting value strictly inside every finite boundary before
    # mapping it to the tail's unconstrained parameterization. In particular,
    # log and logit maps send otherwise valid boundary values to ±Inf.
    Tθ = promote_type(typeof(float(θ0_val)), typeof(float(lo)), typeof(float(hi)))
    loT, hiT = Tθ(lo), Tθ(hi)
    inward(x) = sqrt(eps(Tθ)) * max(one(Tθ), abs(x))
    lo_start = isfinite(loT) ? loT + inward(loT) : -Tθ(1e16)
    hi_start = isfinite(hiT) ? hiT - inward(hiT) : Tθ(1e16)
    θ0_clamped = clamp(Tθ(θ0_val), lo_start, hi_start)
    θ0 = (; θ=θ0_clamped)
    α0 = _unbound_params(ConcreteCT, d, θ0)
    all(isfinite, α0) || throw(ArgumentError("MLE start must map to finite unbounded parameters"))
    cop(α) = ExtremeValueCopula{d}(TT(_rebound_params(ConcreteCT, d, α)...))
    f(α) = -Distributions.loglikelihood(cop(α), U)
    res = try
        Optim.optimize(f, α0, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())
    catch
        Optim.optimize(f, α0, Optim.NelderMead())
    end
    θ̂ = _rebound_params(ConcreteCT, d, Optim.minimizer(res))
    return ExtremeValueCopula{d}(TT(θ̂...)), (; θ̂=θ̂, optimizer=Optim.summary(res),
                        xtol=xtol, converged=Optim.converged(res),
                        iterations=Optim.iterations(res))
end
