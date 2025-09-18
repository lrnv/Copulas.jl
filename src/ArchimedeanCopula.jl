"""
    ArchimedeanCopula{d, TG} <: Copula{d}

`d`-dimensional Archimedean copula with generator `G::TG <: Generator`.

Definition
For an Archimedean generator ``\\phi: [0,\\infty) \\to [0,1]`` that is *d*-monotone (admits a Williamson *d*-transform), the copula is

```math
C(\\mathbf u) = \\phi\\Big( \\phi^{-1}(u_1)+\\cdots+\\phi^{-1}(u_d) \\Big).
```

Fields
    * `G::TG` – generator instance (must implement at least `ϕ(G,t)` and `ϕ⁻¹(G,u)`; optionally higher derivatives `ϕ⁽ᵏ⁾`).

Constructor
    * `ArchimedeanCopula(d::Int, G::Generator)` (asserts `d ≤ max_monotony(G)`).

Sampling
        1. Radial / simplex decomposition via the Williamson transform of ``\\phi`` (default).
    2. Frailty-based shortcut if `G <: AbstractFrailtyGenerator`.

Named families
For each exported `NamedGenerator`, the alias
`NamedCopula{d,...} = ArchimedeanCopula{d, NamedGenerator{...}}` is provided.

# Internal CDF implementation (called by generic `cdf`).
C = ArchimedeanCopula(3, MyGenerator())
# Specialized logpdf (generic interface documented at higher level).

See also: [`Copula`](@ref), [`Generator`](@ref), [`WilliamsonGenerator`](@ref), [`FrailtyGenerator`](@ref), [`ConditionalCopula`](@ref).

References
* [williamson1955multiply](@cite) Williamson (1956), Multiply monotone functions and their Laplace transforms.
* [mcneil2009](@cite) McNeil & Nešlehová (2009), Multivariate Archimedean copulas, d-monotone functions and ℓ₁-norm symmetric distributions.
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

# Log-density using the classical formula
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

# Recover underlying generator type (internal helper).
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

# Conditioning colocated
# Internal helper building distortion for conditional distribution.
function DistortionFromCop(C::ArchimedeanCopula, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p}
    @assert length(js) == length(uⱼₛ)
    T = eltype(uⱼₛ)
    sJ = zero(T)
    @inbounds for u in uⱼₛ
        sJ += ϕ⁻¹(C.G, u)
    end
    return ArchimedeanDistortion(C.G, p, float(sJ), float(T(ϕ⁽ᵏ⁾(C.G, Val{p}(), sJ))))
end

# Conditional copula specialization: remains Archimedean with a tilted generator
function ConditionalCopula(C::ArchimedeanCopula{D}, ::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D, p}
    return ArchimedeanCopula(D - p, TiltedGenerator(C.G, Val{p}(), sum(ϕ⁻¹.(C.G, uⱼₛ))))
end

# Subsetting colocated
SubsetCopula(C::ArchimedeanCopula{d,TG}, dims::NTuple{p, Int}) where {d,TG,p} = ArchimedeanCopula(length(dims), C.G)