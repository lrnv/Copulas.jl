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
    function ArchimedeanCopula(d::Int,G::Generator)
        @assert d <= max_monotony(G) "The generator $G you provided is not $d-monotonous since it has max monotonicity $(max_monotony(G)), and thus this copula does not exists."
        return new{d,typeof(G)}(G)
    end
    ArchimedeanCopula(d::Int, ::IndependentGenerator) = IndependentCopula(d)
    ArchimedeanCopula(d::Int, ::MGenerator) = MCopula(d)
    ArchimedeanCopula(d::Int, ::WGenerator) = WCopula(d)
    ArchimedeanCopula{d,TG}(θ) where {d, TG} = ArchimedeanCopula(d, TG(θ))
end
Distributions.params(C::ArchimedeanCopula) = Distributions.params(C.G) # by default the parameter is the generator's parameters. 



_cdf(C::ArchimedeanCopula, u) = ϕ(C.G, sum(ϕ⁻¹.(C.G, u)))
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .< u .< 1)
        return eltype(u)(-Inf)
    end
    return log(max(ϕ⁽ᵏ⁾(C.G, Val{d}(), sum(ϕ⁻¹.(C.G, u))) * prod(ϕ⁻¹⁽¹⁾.(C.G, u)), 0))
end

# function τ(C::ArchimedeanCopula{d, TG}) where {d, TG}
#     return 4*Distributions.expectation(r -> ϕ(C.G,r), williamson_dist(C.G, Val{d}())) - 1
# end

# Rand function: the default case is williamson
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
# but if frailty is available, use it. 
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
        return S3.parameters[2].name.wrapper
    catch e
        @error "There is no generator type associated with the archimedean type $S"
    end
end

function Distributions.fit(::Type{CT},u) where {CT <: ArchimedeanCopula} 
    # @info "Archimedean fits are by default through inverse kendall tau."
    d = size(u,1)
    τ = StatsBase.corkendall(u')
    # Then the off-diagonal elements of the matrix should be averaged:
    avgτ = (sum(τ) .- d) / (d^2-d)
    GT = generatorof(CT)
    θ = τ⁻¹(GT,avgτ)
    return ArchimedeanCopula(d,GT(θ))
end

function τ(C::ArchimedeanCopula{d,TG}) where {d,TG}
    if applicable(Copulas.τ, C.G)
        return τ(C.G)
    else
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
        Cᵢⱼ = zero(eltype(u))
        for j in 2:d
            Cᵢⱼ += ϕ⁻¹(C.G, U[j - 1, i])
            if iszero(Cᵢⱼ)
                U[j, i] = one(Cᵢⱼ)
            elseif !isfinite(Cᵢⱼ)
                U[j,i] = zero(Cᵢⱼ)
            else
                Dᵢⱼ = ϕ⁽ᵏ⁾(C.G, Val{j - 1}(), Cᵢⱼ) * u[j,i]
                R = ϕ⁽ᵏ⁾⁻¹(C.G, Val{j - 1}(), Dᵢⱼ; start_at=Cᵢⱼ)
                U[j, i] = ϕ(C.G, R - Cᵢⱼ)
            end
        end
    end
    return U
end

# Conditioning colocated
function DistortionFromCop(C::ArchimedeanCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p}
    @assert length(js) == length(uⱼₛ)
    sJ = zero(eltype(uⱼₛ))
    @inbounds for u in uⱼₛ
        sJ += ϕ⁻¹(C.G, float(u))
    end
    return ArchimedeanDistortion(C.G, p, float(sJ), ϕ⁽ᵏ⁾(C.G, Val{p}(), sJ))
end

# Conditional copula specialization: remains Archimedean with a tilted generator
function ConditionalCopula(C::ArchimedeanCopula{D}, ::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D, p}
    return ArchimedeanCopula(D - p, TiltedGenerator(C.G, Val{p}(), sum(ϕ⁻¹.(C.G, uⱼₛ))))
end

# Subsetting colocated
SubsetCopula(C::ArchimedeanCopula{d,TG}, dims::NTuple{p, Int}) where {d,TG,p} = ArchimedeanCopula(length(dims), C.G)