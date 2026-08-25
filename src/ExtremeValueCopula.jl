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

ExtremeValueCopula(d::Int, tail::Tail) = ExtremeValueCopula{d}(tail)

@inline _ev_encoded_dimension(CT) = Base.unwrap_unionall(CT).parameters[1]

function _typed_extreme_value(CT, d::Int, args...; kwargs...)
    encoded = _ev_encoded_dimension(CT)
    if !(encoded isa TypeVar) && encoded != d
        throw(DimensionMismatch(
            "encoded dimension d=$encoded does not match requested dimension $d",
        ))
    end
    return ExtremeValueCopula{d}(tailof(CT)(args...; kwargs...))
end

# Canonical constructor: FamilyCopula{d}(params...).
#
# If `d` is not encoded in the type, non-structured parameterizations must use
# the runtime convenience form FamilyCopula(d, params...). Structured family
# constructors may provide more specific methods that infer `d` from a matrix
# or vector.
function (CT::Type{<:ExtremeValueCopula{D}})(args...; kwargs...) where {D}
    d = _ev_encoded_dimension(CT)
    d isa TypeVar && throw(ArgumentError(
        "the copula dimension must be specified as FamilyCopula{d}(...) " *
        "or FamilyCopula(d, ...)",
    ))
    return _typed_extreme_value(CT, d, args...; kwargs...)
end

# Resolve the only generic intersection left by integer-valued parameters:
# for FamilyCopula{d}(first::Int, ...), `first` is a parameter; for the
# unparameterized FamilyCopula(first::Int, ...), it is the runtime dimension.
function (CT::Type{<:ExtremeValueCopula{D}})(first::Int, args...; kwargs...) where {D}
    d = _ev_encoded_dimension(CT)
    if d isa TypeVar
        return _typed_extreme_value(CT, first, args...; kwargs...)
    end
    return _typed_extreme_value(CT, d, first, args...; kwargs...)
end

# Runtime-dimension sugar for an unparameterized family alias:
#     FamilyCopula(d, params...)
#
# This deliberately stays separate from the canonical encoded-dimension route
# above. In particular, FamilyCopula{d}(integer_parameter, ...) always treats
# the integer as a model parameter.
(CT::Type{<:ExtremeValueCopula})(d::Int, args...; kwargs...) =
    _typed_extreme_value(CT, d, args...; kwargs...)

# Generic fitting reconstructs from concrete `typeof(C)` values. It should not
# call the public runtime constructor on a fully concrete FamilyCopula{d,T},
# because that would make an integer model parameter indistinguishable from d.
_construct_from_params(
    CT::Type{<:ExtremeValueCopula},
    d::Int,
    args...;
    kwargs...,
) = _typed_extreme_value(CT, d, args...; kwargs...)

function _cdf(C::ExtremeValueCopula{2,<:BivariatePickandsTail}, u)
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

_cdf(C::ExtremeValueCopula{d, TT}, u) where {d, TT} = exp(-ℓ(C.tail, .- log.(u)))
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
    return -val + log(core) + x + y
end

Distributions._logpdf(C::ExtremeValueCopula{2,<:BivariatePickandsTail}, u) =
    _bivariate_pickands_logpdf(C, u)

# Generic d-dimensional density from the mixed STDF partials and the
# partition formula for absolutely continuous extreme-value copulas.
function Distributions._logpdf(C::ExtremeValueCopula{d}, u) where {d}
    all(ui -> zero(ui) < ui <= one(ui), u) || return oftype(float(first(u)), -Inf)
    x = -log.(u)
    val = ℓ(C.tail, x)
    any(isone, u) && return oftype(val, -Inf)
    logpos = logneg = oftype(val, -Inf)
    partials = Dict{Tuple{Vararg{Int}},Tuple{Int,typeof(val)}}()

    for π in Combinatorics.partitions(collect(1:d))
        sgn = isodd(d + length(π)) ? -1 : 1
        logabs = zero(val)
        nonzero = true
        for block in π
            I = Tuple(block)
            blocksgn, blocklog = get!(partials, I) do
                _ellpartial_signlog(C.tail, x, I)
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
    return -val + sum(x) + logpos
end
τ(C::ExtremeValueCopula{2}) = QuadGK.quadgk(t -> d²A(C.tail, t) * t * (1 - t) / max(A(C.tail, t), _δ(t)), 0.0, 1.0)[1]
ρ(C::ExtremeValueCopula{2}) = 12 * QuadGK.quadgk(t -> 1 / (1 + A(C.tail, t))^2, 0.0, 1.0)[1] - 3
β(C::ExtremeValueCopula{2}) = 4^(1 - A(C.tail, 0.5)) - 1
λᵤ(C::ExtremeValueCopula{2}) = 2 * (1 - A(C.tail, 0.5))
λₗ(C::ExtremeValueCopula{2}) =  A(C.tail, 0.5) > 0.5 ? 0.0 : 1.0
function τ⁻¹(::Type{T},τ_val) where {T<:ExtremeValueCopula{2}}
    return τ⁻¹(tailof(T),τ_val)
end


# Sampling is selected by tail capability. Families with a preferable exact
# sampler may still specialize `_rand!` for their concrete copula type.
function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch(
        "output dimension does not match copula dimension",
    ))
    return _rand_tail!(rng, C.tail, X)
end

function _rand_tail!(
    rng::Distributions.AbstractRNG,
    tail::BivariatePickandsTail,
    X::AbstractMatrix{T},
) where {T<:Real}
    E = ExtremeDist(tail)
    for i in axes(X, 2)
        z = rand(rng, E)
        w = rand(rng) < _ghoudi_mixture_probability(tail, z) ? rand(rng) : rand(rng) * rand(rng)
        a = A(tail, z)
        X[1, i] = exp(log(w) * z / a)
        X[2, i] = exp(log(w) * (1 - z) / a)
    end
    return X
end

DistortionFromCop(C::ExtremeValueCopula{2, TT}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) where TT = BivEVDistortion(C.tail, Int8(js[1]), float(uⱼₛ[1]))

# Fitting functions: the default one is in the EmpiricalEvTail because this is what will happen by default.
# For this moment generic mle works... maybe we could be implement others specifyc methods maybe upper and lower tail

# # Parametric-type constructors to allow generic fit to reconstruct from NamedTuple params
# function (::Type{ExtremeValueCopula{D, TT}})(d::Integer, θ::NamedTuple) where {D, TT<:Tail}
#     d == D || @warn "Dimension mismatch constructing ExtremeValueCopula: got d=$(d), type encodes D=$(D). Proceeding with d."
#     # Get parameter order from an example of the tail
#     Tex = _example(ExtremeValueCopula{D, TT}, D).tail
#     names = collect(keys(Distributions.params(Tex)))
#     # Support both plain names and optional tail_-prefixed names
#     getp(nt::NamedTuple, k::Symbol) = haskey(nt, k) ? nt[k] : (haskey(nt, Symbol(:tail_, k)) ? nt[Symbol(:tail_, k)] : throw(ArgumentError("Missing parameter $(k) for ExtremeValueCopula.")))
#     vals = map(n -> getp(θ, n), names)
#     return ExtremeValueCopula(d, TT(vals...))
# end
# function (::Type{ExtremeValueCopula{D, TT}})(d::Integer; kwargs...) where {D, TT<:Tail}
#     return (ExtremeValueCopula{D, TT})(d, NamedTuple(kwargs))
# end
tailof(S::Type{<:ExtremeValueCopula}) = fieldtype(S, :tail)

##############################################################################################################################
####### Fitting functions for univariate tails only (Extreme Value Copulas).
##############################################################################################################################

_example(CT::Type{<:ExtremeValueCopula}, d) = _typed_extreme_value(CT, d; _rebound_params(CT, d, fill(0.01, fieldcount(tailof(CT))))...)
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
    θ = m isa Val{:itau} ? τ⁻¹(CT,  StatsBase.corkendall(U')[1,2]) :
        m isa Val{:irho} ? ρ⁻¹(CT,  StatsBase.corspearman(U')[1,2]) :
                           β⁻¹(CT,  corblomqvist(U')[1,2])
    lo, hi = _θ_bounds(tailof(CT), 2)
    # unbounded limits are bound to 1e16 (inf) and zero is bound to (1e-16) for stability
    θ = clamp(θ, iszero(lo) ? 1e-16 : lo, isinf(hi) ? 1e16 : hi)
    return _typed_extreme_value(CT, 2, θ), (; θ̂=(θ=θ,))
end
function _fit(CT::Type{<:ExtremeValueCopula{d, GT} where {d, GT<:OneParameterPickandsTail}}, U, ::Val{:iupper})
    θ = clamp(λᵤ⁻¹(CT, λᵤ(U)), _θ_bounds(tailof(CT), 2)...)
    return _typed_extreme_value(CT, 2, θ), (; θ̂=(θ=θ,))
end

function _fit(CT::Type{<:ExtremeValueCopula{d, GT} where {d, GT<:OneParameterPickandsTail}}, U, ::Val{:mle}; start::Union{Symbol,Real}=:itau, xtol::Real=1e-8)
    d = size(U,1)
    TT = tailof(CT)
    lo, hi = _θ_bounds(TT, d)
    θ0_val = if start isa Real
        start
    else
        initial_params = start ∈ (:itau, :irho, :ibeta, :iupper) ? _fit(CT, U, Val{start}())[2].θ̂ : only(Distributions.params(_example(CT, d)))
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
    α0 = _unbound_params(CT, d, θ0)
    all(isfinite, α0) || throw(ArgumentError("MLE start must map to finite unbounded parameters"))
    cop(α) = _typed_extreme_value(CT, d, _rebound_params(CT, d, α)...)
    f(α) = -Distributions.loglikelihood(cop(α), U)
    res = try
        Optim.optimize(f, α0, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())
    catch
        Optim.optimize(f, α0, Optim.NelderMead())
    end
    θ̂ = _rebound_params(CT, d, Optim.minimizer(res))
    return _typed_extreme_value(CT, d, θ̂...), (; θ̂=θ̂, optimizer=Optim.summary(res),
                        xtol=xtol, converged=Optim.converged(res),
                        iterations=Optim.iterations(res))
end
