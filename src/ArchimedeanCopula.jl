"""
    ArchimedeanCopula{d, TG}

Fields:
    - G::TG : the generator <: Generator.

Constructor:

    ArchimedeanCopula(d::Int,G::Generator)

For some Archimedean [`Generator`](@ref) `G::Generator` and some dimenson `d`, this class models the archimedean copula which has this generator. The constructor checks for validity by ensuring that `max_monotony(G) ≥ d`. The ``d``-variate archimedean copula with generator ``\\phi`` writes:

```math
C(\\mathbf u) = \\phi^{-1}\\left(\\sum_{i=1}^d \\phi(u_i)\\right)
```

The default sampling method is the Radial-simplex decomposition using the Williamson transformation of ``\\phi``.

There exists several known parametric generators that are implement in the package. For every `NamedGenerator <: Generator` implemented in the package, we provide a type alias ``NamedCopula{d,...} = ArchimedeanCopula{d,NamedGenerator{...}}` to be able to manipulate the classic archimedean copulas without too much hassle for known and usefull special cases.

A generic archimedean copula can be constructed as follows:

```julia
struct MyGenerator <: Generator end
ϕ(G::MyGenerator,t) = exp(-t) # your archimedean generator, can be any d-monotonous function.
max_monotony(G::MyGenerator) = Inf # could depend on generators parameters.
C = ArchimedeanCopula(d,MyGenerator())
```

The obtained model can be used as follows:
```julia
spl = rand(C,1000)   # sampling
cdf(C,spl)           # cdf
pdf(C,spl)           # pdf
loglikelihood(C,spl) # llh
```

Bonus: If you know the Williamson d-transform of your generator and not your generator itself, you may take a look at [`WilliamsonGenerator`](@ref) that implements them. If you rather know the frailty distribution, take a look at `WilliamsonFromFrailty`.

References:
* [williamson1955multiply](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
"""
struct ArchimedeanCopula{d,TG} <: Copula{d}
    G::TG
    function ArchimedeanCopula(d::Int, G::Generator)
        @assert d <= max_monotony(G) "The generator $G you provided is not $d-monotonous since it has max monotonicity $(max_monotony(G)), and thus this copula does not exists."
        return new{d,typeof(G)}(G)
    end
end

# Constructors: 
ArchimedeanCopula(d::Int, ::IndependentGenerator) = IndependentCopula(d)
ArchimedeanCopula(d::Int, ::MGenerator) = MCopula(d)
ArchimedeanCopula(d::Int, ::WGenerator) = WCopula(d)

ArchimedeanCopula{d,TG}(args...; kwargs...) where {d, TG} = ArchimedeanCopula(d, TG(args...; kwargs...))
ArchimedeanCopula{D,TG}(d::Int, args...; kwargs...) where {D, TG} = ArchimedeanCopula{d,TG}(args...; kwargs...)
(CT::Type{<:ArchimedeanCopula{D, <:Generator} where D})(d::Int, args...; kwargs...) = ArchimedeanCopula(d, generatorof(CT)(args...; kwargs...))

Distributions.params(C::ArchimedeanCopula) = Distributions.params(C.G) # by default the parameter is the generator's parameters. 

_cdf(C::ArchimedeanCopula, u) = ϕ(C.G, sum(ϕ⁻¹.(C.G, u)))
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .< u .< 1)
        return eltype(u)(-Inf)
    end
    return log(max(ϕ⁽ᵏ⁾(C.G, Val{d}(), sum(ϕ⁻¹.(C.G, u))) * prod(ϕ⁻¹⁽¹⁾.(C.G, u)), 0))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d, TG}, x::AbstractVector{T}) where {T<:Real, d, TG}
    # By default, we use the Williamson sampling.
    Random.randexp!(rng,x)
    r = rand(rng, williamson_dist(C.G, Val{d}()))
    sx = sum(x)
    for i in 1:length(C)
        x[i] = ϕ(C.G,r * x[i]/sx)
    end
    return x
end 
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d, GT}, x::AbstractVector{T}) where {T<:Real, d, GT<:AbstractFrailtyGenerator}
    F = frailty(C.G)
    Random.randexp!(rng, x)
    f = rand(rng, F)
    x .= ϕ.(C.G, x ./ f)
    return x
end

function generatorof(::Type{S}) where {S <: ArchimedeanCopula}
    S2 = hasproperty(S,:body) ? S.body : S
    S3 = hasproperty(S2, :body) ? S2.body : S2
    try
        S4 = S3.parameters[2]
        return hasproperty(S4, :name) ? S4.name.wrapper : S4
    catch e
        @error "There is no generator type associated with the archimedean type $S"
    end
end
generatorof(::Type{ArchimedeanCopula{d,GT}}) where {d,GT} = GT #this solution problem...


function τ(C::ArchimedeanCopula{d,TG}) where {d,TG}
    if applicable(Copulas.τ, C.G)
        return τ(C.G)
    else
        # 4*Distributions.expectation(r -> ϕ(C.G,r), williamson_dist(C.G, Val{d}())) - 1
        return @invoke τ(C::Copula)
    end
end
function τ⁻¹(::Type{T},τ_val) where {T<:ArchimedeanCopula}
    return τ⁻¹(generatorof(T),τ_val)
end
function rosenblatt(C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}) where {d,TG}
    @assert d == size(u, 1)
    U = zero(u)
    for i in axes(u,2)
        U[1, i] = u[1, i]
        rⱼ₋₁ = zero(eltype(u))
        rⱼ = ϕ⁻¹(C.G, u[1,i])
        for j in 2:d
            rⱼ₋₁ = rⱼ
            if !isfinite(rⱼ₋₁)
                U[j,i] = one(rⱼ)
            else
                rⱼ += ϕ⁻¹(C.G, u[j,i])
                if iszero(rⱼ)
                     U[j,i] = zero(rⱼ)
                else
                    A, B = ϕ⁽ᵏ⁾(C.G, Val(j - 1), rⱼ), ϕ⁽ᵏ⁾(C.G, Val(j - 1), rⱼ₋₁)
                    U[j,i] = A / B
                end
            end
        end
    end
    return U
end
function inverse_rosenblatt(C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}) where {d,TG}
    @assert d == size(u, 1)
    U = zero(u)
    for i in axes(u, 2)
        U[1,i] = u[1,i]
        Cᵢⱼ = ϕ⁻¹(C.G, U[1,i])
        for j in 2:d
            if iszero(Cᵢⱼ)
                U[j, i] = one(Cᵢⱼ)
            elseif !isfinite(Cᵢⱼ)
                U[j,i] = zero(Cᵢⱼ)
            else
                Dᵢⱼ = ϕ⁽ᵏ⁾(C.G, Val{j - 1}(), Cᵢⱼ) * u[j,i]
                R = ϕ⁽ᵏ⁾⁻¹(C.G, Val{j - 1}(), Dᵢⱼ; start_at=Cᵢⱼ)
                U[j, i] = ϕ(C.G, R - Cᵢⱼ)
                Cᵢⱼ = R
            end
        end
    end
    return U
end

function DistortionFromCop(C::ArchimedeanCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p}
    @assert length(js) == length(uⱼₛ)
    T = eltype(uⱼₛ)
    sJ = zero(T)
    @inbounds for u in uⱼₛ
        sJ += ϕ⁻¹(C.G, u)
    end
    return ArchimedeanDistortion(C.G, p, float(sJ), float(T(ϕ⁽ᵏ⁾(C.G, Val{p}(), sJ))))
end
function ConditionalCopula(C::ArchimedeanCopula{D}, ::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D, p}
    return ArchimedeanCopula(D - p, TiltedGenerator(C.G, Val{p}(), sum(ϕ⁻¹.(C.G, uⱼₛ))))
end
SubsetCopula(C::ArchimedeanCopula{d,TG}, dims::NTuple{p, Int}) where {d,TG,p} = ArchimedeanCopula(length(dims), C.G)

##############################################################################################################################
####### Fitting interfaces. 
##############################################################################################################################

_example(::Type{ArchimedeanCopula}, d) = throw("Cannot fit an Archimedean copula without specifying its generator (unless you set method=:gnz2011)")
_example(CT::Type{<:ArchimedeanCopula}, d) = CT(d; _rebound_params(CT, d, fill(0.01, fieldcount(generatorof(CT))))...)
_example(::Type{<:ArchimedeanCopula{d,<:WilliamsonGenerator{d2, TX}} where {d,d2, TX}}, d) = ArchimedeanCopula(d,i𝒲(Distributions.MixtureModel([Distributions.Dirac(1), Distributions.Dirac(2)]),d))
_example(::Type{<:ArchimedeanCopula{d,<:FrailtyGenerator} where {d}}, d) = throw("No default example for frailty geenrators are implemented")

_unbound_params(CT::Type{<:ArchimedeanCopula}, d, θ) = _unbound_params(generatorof(CT), d, θ)
_rebound_params(CT::Type{<:ArchimedeanCopula}, d, α) = _rebound_params(generatorof(CT), d, α)

_available_fitting_methods(::Type{ArchimedeanCopula}) = (:gnz2011,)
_available_fitting_methods(::Type{<:ArchimedeanCopula{d,GT} where {d,GT<:Generator}}) = (:mle,)
_available_fitting_methods(::Type{<:ArchimedeanCopula{d,GT} where {d,GT<:UnivariateGenerator}}) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(::Type{<:ArchimedeanCopula{d,<:WilliamsonGenerator{d2, TX}} where {d,d2, TX}}) = Tuple{}() # No fitting method. 
_available_fitting_methods(::Type{<:ArchimedeanCopula{d,<:WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}} where {d,d2}}) = (:gnz2011,)


function _fit(::Union{Type{ArchimedeanCopula},Type{<:ArchimedeanCopula{d,<:WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}} where {d,d2}}}, U, ::Val{:gnz2011})
    # When fitting only an archimedean copula with no specified general, you get and empiricalgenerator fitted. 
    return ArchimedeanCopula(size(U, 1), EmpiricalGenerator(U)), (;)
end

function _fit(CT::Type{<:ArchimedeanCopula{d, GT} where {d, GT<:UnivariateGenerator}}, U, m::Union{Val{:itau},Val{:irho}})
    d = size(U,1)
    GT = generatorof(CT)
    
    f = m isa Val{:itau} ?  StatsBase.corkendall :  StatsBase.corspearman
    invf =  m isa Val{:itau} ?  τ⁻¹ : ρ⁻¹

    θs = map(v -> invf(GT, clamp(v, -1, 1)), _uppertriangle_stats(f(U')))
    θ = clamp(Statistics.mean(θs), _θ_bounds(GT, d)...)
    return CT(d, θ), (; θ̂=θ, eps)
end
function _fit(CT::Type{<:ArchimedeanCopula{d, GT} where {d, GT<:UnivariateGenerator}}, U, ::Val{:ibeta})
    d    = size(U,1); δ = 1e-8; GT = generatorof(CT)
    βobs = clamp(β(U), -1+1e-10, 1-1e-10)
    lo,hi = _θ_bounds(GT,d)
    fβ(θ) = β(CT(d,θ))
    a0 = isfinite(lo) ? lo+δ : -5.0 ; b0 = isfinite(hi) ? hi-δ :  5.0
    βmin, βmax = fβ(a0), fβ(b0)
    if βmin > βmax; βmin, βmax = βmax, βmin; end
    θ = βobs ≤ βmin ? a0 : βobs ≥ βmax ? b0 : Roots.find_zero(θ -> fβ(θ)-βobs, (a0,b0), Roots.Brent(); xatol=1e-8, rtol=0)
    return CT(d,θ), (; θ̂=θ)
end

function _fit(CT::Type{<:ArchimedeanCopula{d, GT} where {d, GT<:UnivariateGenerator}}, U, ::Val{:mle}; start::Union{Symbol,Real}=:itau, xtol::Real=1e-8)
    d = size(U,1)
    GT = generatorof(CT)
    lo, hi = _θ_bounds(GT, d)
    θ₀ = [(lo+hi)/2]
    if start isa Real 
        θ₀[1] = start
    elseif start ∈ (:itau, :irho)
        try 
            θ₀[1] = only(Distributions.params(_fit(CT, U, Val{start}())[1]))
        catch e
        end
    end
    θ₀[1] = clamp(θ₀[1], lo, hi)
    f(θ) = -Distributions.loglikelihood(CT(d, θ[1]), U)
    res = Optim.optimize(f, lo, hi,  θ₀, Optim.Fminbox(Optim.LBFGS()), autodiff = :forward)
    θ̂     = Optim.minimizer(res)[1]
    return CT(d, θ̂), (; θ̂=θ̂, optimizer=:GradientDescent,
                        xtol=xtol, converged=Optim.converged(res), 
                        iterations=Optim.iterations(res))
end