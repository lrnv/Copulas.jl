"""
    ExtremeValueCopula{d, TT}

Constructor

    ExtremeValueCopula(d, tail::Tail)

Extreme-value copulas model tail dependence via a stable tail dependence function (STDF) ``\\ell`` or, equivalently,
via a Pickands dependence function ``A``. In any dimension ``d``, the copula cdf is

```math
\\displaystyle C(u) = \\\exp\\!\\left(-\\, \\\ell(-\\log u_1,\\ldots,-\\log u_d) \\right).
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
struct ExtremeValueCopula{d, TT<:Tail} <: Copula{d}
    tail::TT
    function ExtremeValueCopula(d, tail::Tail)
        @assert _is_valid_in_dim(tail, d)
        return new{d, typeof(tail)}(tail)
    end
end
_cdf(C::ExtremeValueCopula{d, TT}, u) where {d, TT} = exp(-ℓ(C.tail, .- log.(u)))
Distributions.params(C::ExtremeValueCopula) = Distributions.params(C.tail)

#### Restriction to bivariate cases of the following methods: 
function Distributions._logpdf(C::ExtremeValueCopula{2, TT}, u) where {TT}
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return -Inf 
    # On the broder, the limit of the pdf is 0 and thus logpdf tends to -Inf
    (u1 == 1.0 || u2 == 1.0) && return -Inf
    x, y = -log(u1), -log(u2)
    val, du, dv, dudv = _biv_der_ℓ(C.tail, (x, y))
    core = -dudv + du*dv
    core ≤ 0 && return -Inf
    return -val + log(core) + x + y
end
τ(C::ExtremeValueCopula{2}) = QuadGK.quadgk(t -> d²A(C.tail, t) * t * (1 - t) / max(A(C.tail, t), _δ(t)), 0.0, 1.0)[1]
ρ(C::ExtremeValueCopula{2}) = 12 * QuadGK.quadgk(t -> 1 / (1 + A(C.tail, t))^2, 0.0, 1.0)[1] - 3
β(C::ExtremeValueCopula{2}) = 4^(1 - A(C.tail, 0.5)) - 1
λᵤ(C::ExtremeValueCopula{2}) = 2 * (1 - A(C.tail, 0.5))
λₗ(C::ExtremeValueCopula{2}) =  A(C.tail, 0.5) > 0.5 ? 0.0 : 1.0

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, TT}, X::DenseMatrix{T}) where {T<:Real, TT}
    # More efficient Matrix sampler: 
    d,n = size(X)
    @assert d==2
    E = ExtremeDist(C.tail)
    for i in 1:n
        z = rand(rng, E)
        w = rand(rng) < _probability_z(C.tail, z) ? rand(rng) : rand(rng) * rand(rng)
        a = A(C.tail, z)
        X[1,i] = exp(log(w)*z/a)
        X[2,i] = exp(log(w)*(1-z)/a)
    end
    return X 
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, TT},
                              x::AbstractVector{T}) where {T<:Real, TT}
    u1, u2 = rand(rng), rand(rng)
    z  = rand(rng, ExtremeDist(C.tail))
    w  = (rand(rng) < _probability_z(C.tail, z)) ? u1 : (u1*u2)
    a  = A(C.tail, z)
    x[1] = exp(log(w)*z/a)
    x[2] = exp(log(w)*(1-z)/a)
    return x
end
DistortionFromCop(C::ExtremeValueCopula{2, TT}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) where TT = BivEVDistortion(C.tail, Int8(js[1]), float(uⱼₛ[1]))



# Fitting functions: the default one is in the EmpiricalEvTail because this is what will happen by default. 
# For this moment generic mle works... maybe we could be implement others specifyc methods maybe upper and lower tail


# Parametric-type constructors to allow generic fit to reconstruct from NamedTuple params
function (::Type{ExtremeValueCopula{D, TT}})(d::Integer, θ::NamedTuple) where {D, TT<:Tail}
    d == D || @warn "Dimension mismatch constructing ExtremeValueCopula: got d=$(d), type encodes D=$(D). Proceeding with d."
    # Get parameter order from an example of the tail
    Tex = _example(ExtremeValueCopula{D, TT}, D).tail
    names = collect(keys(Distributions.params(Tex)))
    # Support both plain names and optional tail_-prefixed names
    getp(nt::NamedTuple, k::Symbol) = haskey(nt, k) ? nt[k] : (haskey(nt, Symbol(:tail_, k)) ? nt[Symbol(:tail_, k)] : throw(ArgumentError("Missing parameter $(k) for ExtremeValueCopula.")))
    vals = map(n -> getp(θ, n), names)
    return ExtremeValueCopula(d, TT(vals...))
end
function (::Type{ExtremeValueCopula{D, TT}})(d::Integer; kwargs...) where {D, TT<:Tail}
    return (ExtremeValueCopula{D, TT})(d, NamedTuple(kwargs))
end

##############################################################################################################################
####### Fitting functions for univarate generators only. 
##############################################################################################################################

# When no generator is specified: 
_example(CT::Type{ExtremeValueCopula}, d) = throw("Cannot fit an Extreme Value copula without specifying its Tail (unless you set method=:ols)")
_unbound_params(CT::Type{ExtremeValueCopula}, d, θ) = throw("Cannot fit an Extreme Value copula without specifying its Tail (unless you set method=:ols)")
_rebound_params(CT::Type{ExtremeValueCopula}, d, α) = throw("Cannot fit an Extreme Value copula without specifying its Tail (unless you set method=:ols)")
_default_method(::Type{<:ExtremeValueCopula}) = :mle
# By default, when no tail is given in the call, we fit the empirical tail. 
_default_method(::Type{ExtremeValueCopula}) = :ols # or cfg or pickands ? 
function _fit(::Type{ExtremeValueCopula}, U, method::Union{Val{:ols}, Val{:cfg}, Val{:pickands}}; pseudo_values = true, grid::Int = 401, eps::Real = 1e-3, kwargs...)
    C = EmpiricalEVCopula(U; 
        method=typeof(method).parameters[1], 
        grid=grid, eps=eps, 
        pseudo_values=pseudo_values, kwargs...)
    return C, (; pseudo_values, grid, eps)
end
function _fit(CT::Type{<:ExtremeValueCopula{d, TT} where d}, U, ::Val{:itau}) where {TT<:UnivariateTail2}
    d = size(U,1)
    # So here we do the mean on the theta side, maybe we should do it on the \tau side ?
    θs   = map(v -> τ⁻¹(TT, clamp(v, -1, 1)), _uppertriangle_stats(StatsBase.corkendall(U')))
    θ = clamp(Statistics.mean(θs), _θ_bounds(TT, d)...)
    return CT(d, θ), (; eps)
end
function _fit(CT::Type{<:ExtremeValueCopula{d, TT} where d}, U, ::Val{:irho})  where {TT<:UnivariateTail2}
    d = size(U,1)
    # So here we do the mean on the theta side, maybe we should do it on the \rho side ?
    θs   = map(v -> ρ⁻¹(TT, clamp(v, -1, 1)), _uppertriangle_stats(StatsBase.corspearman(U')))
    θ = clamp(Statistics.mean(θs), _θ_bounds(TT, d)...)
    return CT(d, θ), (; eps)
end
function _fit(CT::Type{<:ExtremeValueCopula{d,TT} where d}, U, ::Val{:ibeta}) where {TT<:UnivariateTail2}
    d    = size(U,1); δ = 1e-8
    βobs = clamp(blomqvist_beta(U), -1+1e-10, 1-1e-10)
    lo,hi = _θ_bounds(TT,d)
    fβ(θ) = β(CT(d,θ))
    a0 = isfinite(lo) ? lo+δ : -5.0 ; b0 = isfinite(hi) ? hi-δ :  5.0
    βmin, βmax = fβ(a0), fβ(b0)
    if βmin > βmax; βmin, βmax = βmax, βmin; end
    θ = if βobs ≤ βmin
        a0
    elseif βobs ≥ βmax
        b0
    else
        Roots.find_zero(θ -> fβ(θ)-βobs, (a0,b0), Roots.Brent(); xatol=1e-8, rtol=0)
    end
    return CT(d,θ), (; θ̂=θ)
end
function _fit(CT::Type{<:ExtremeValueCopula{d,TT} where d}, U, ::Val{:iupper}) where {TT<:UnivariateTail2}
    d   = size(U,1); δ = 1e-8
    λobs = clamp(upper_tail(U; est=:log), 0.0, 1.0)  # empirical upper tail (EV)
    lo, hi = _θ_bounds(TT, d)
    f(θ) = λᵤ(CT(d, θ))
    a = isfinite(lo) ? lo + δ : -5.0
    b = isfinite(hi) ? hi - δ :  5.0
    λa, λb = f(a), f(b)
    θ = λobs ≤ min(λa,λb) ? (λa ≤ λb ? a : b) :
        λobs ≥ max(λa,λb) ? (λa ≥ λb ? a : b) :
        Roots.find_zero(θ -> f(θ) - λobs, (a,b), Roots.Brent(); xatol=1e-8, rtol=0)
    return CT(d, θ), (; θ̂=θ)
end

function _fit(CT::Type{<:ExtremeValueCopula{d, TT} where d}, U, ::Val{:mle}; start::Union{Symbol,Real}=:ibeta, xtol::Real=1e-8)  where {TT<:UnivariateTail2}
    d = size(U,1)
    lo, hi = _θ_bounds(TT, d)
    θ0 = start isa Real ? start : 
         start ∈ (:itau, :irho) ? only(Distributions.params(_fit(CT, U, Val{start}())[1])) : 
         throw("You imputed start=$start, while i require either a real number, :itau or :irho")
    θ0 = clamp(θ0, lo, hi)
    f(θ) = -Distributions.loglikelihood(CT(d, θ[1]), U)
    res = Optim.optimize(f, lo, hi,  [θ0], Optim.Fminbox(Optim.LBFGS()), autodiff = :forward)
    θ̂     = Optim.minimizer(res)[1]
    return CT(d, θ̂), (; θ̂=θ̂, optimizer=:GradientDescent,
                        xtol=xtol, converged=Optim.converged(res), 
                        iterations=Optim.iterations(res))
end