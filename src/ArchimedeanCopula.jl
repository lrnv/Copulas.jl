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

A generic archimdean copula can be constructed as follows:

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
        @assert d <= max_monotony(G) "The generator you provided is not d-monotonous according to its max_monotonicity property, and thus this copula does not exists."
        return new{d,typeof(G)}(G)
    end
end
ϕ(C::ArchimedeanCopula{d,TG},t)      where {d,TG} = ϕ(C.G, t)
ϕ⁻¹(C::ArchimedeanCopula{d,TG},t)    where {d,TG} = ϕ⁻¹(C.G, t)
ϕ⁻¹⁽¹⁾(C::ArchimedeanCopula{d,TG},t) where {d,TG} = ϕ⁻¹⁽¹⁾(C.G, t)
ϕ⁽¹⁾(C::ArchimedeanCopula{d,TG},t)   where {d,TG} = ϕ⁽¹⁾(C.G, t)
ϕ⁽ᵏ⁾(C::ArchimedeanCopula{d,TG},k,t) where {d,TG} = ϕ⁽ᵏ⁾(C.G, k, t)
williamson_dist(C::ArchimedeanCopula{d,TG}) where {d,TG} = williamson_dist(C.G, Val{d}())

_cdf(C::ArchimedeanCopula, u) = ϕ(C, sum(ϕ⁻¹.(C, u)))
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .< u .< 1)
        return eltype(u)(-Inf)
    end
    return log(ϕ⁽ᵏ⁾(C, Val{d}(), sum(ϕ⁻¹.(C, u))) * prod(ϕ⁻¹⁽¹⁾.(C, u)))
end

function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG<:ClaytonGenerator}
    if !all(0 .< u .< 1) || (C.G.θ < 0 && sum(u .^ -(C.G.θ)) < (d - 1))
        return eltype(u)(-Inf)
    end
    return log(ϕ⁽ᵏ⁾(C, Val{d}(), sum(ϕ⁻¹.(C, u))) * prod(ϕ⁻¹⁽¹⁾.(C, u)))
end

# function τ(C::ArchimedeanCopula)
#     return 4*Distributions.expectation(r -> ϕ(C,r), williamson_dist(C)) - 1
# end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # By default, we use the Williamson sampling.
    Random.randexp!(rng,x)
    r = rand(rng,williamson_dist(C))
    sx = sum(x)
    for i in 1:length(C)
        x[i] = ϕ(C,r * x[i]/sx)
    end
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # More efficient version that precomputes the Williamson transform on each call to sample in batches:
    Random.randexp!(rng,A)
    n = size(A,2)
    r = rand(rng,williamson_dist(C),n)
    for i in 1:n
        sx = sum(A[:,i])
        for j in 1:length(C)
            A[j,i] = ϕ(C,r[i] * A[j,i]/sx)
        end
    end
    return A
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

τ(C::ArchimedeanCopula{d,TG}) where {d,TG} = τ(C.G)
function τ⁻¹(::Type{T},τ_val) where {T<:ArchimedeanCopula}
    return τ⁻¹(generatorof(T),τ_val)
end

################################################################################################
################                                                                ################
################ Define all "named" archimedean copulas at once like that :     ################
################                                                                ################
################################################################################################


## Automatic syntactic sugar for all ZeroVariateGenerators and UnivariateGenerators.
## see https://discourse.julialang.org/t/how-to-dispatch-on-a-type-alias/106476/38?u=lrnv
function generatorof(::Type{S}) where {S <: ArchimedeanCopula}
    S2 = hasproperty(S,:body) ? S.body : S
    S3 = hasproperty(S2, :body) ? S2.body : S2
    try
        return S3.parameters[2].name.wrapper
    catch e
        @error "There is no generator type associated with the archimedean type $S"
    end
end
for T in InteractiveUtils.subtypes(ZeroVariateGenerator)
    G = Symbol(last(split(string(T),'.')))
    C = Symbol(string(G)[begin:end-9]*"Copula")
    @eval begin
        const ($C){d} = ArchimedeanCopula{d,($G)}
        ($C)(d) = ArchimedeanCopula(d,($G)())
    end
end
for T in InteractiveUtils.subtypes(UnivariateGenerator)
    G = Symbol(last(split(string(T),'.')))
    C = Symbol(string(G)[begin:end-9]*"Copula")
    @eval begin
        const ($C){d,Tθ} = ArchimedeanCopula{d,($G){Tθ}}
        ($C)(d,θ) = ArchimedeanCopula(d,($G)(θ))
    end
end

# The zero-variate ones just need a few more methods:
Distributions._logpdf(::ArchimedeanCopula{d,IndependentGenerator}, u) where {d} = all(0 .<= u .<= 1) ? zero(eltype(u)) : eltype(u)(-Inf)
Distributions._logpdf(::ArchimedeanCopula{d,MGenerator},           u) where {d} = all(u == u[1]) ?     zero(eltype(u)) : eltype(u)(-Inf)
Distributions._logpdf(::ArchimedeanCopula{d,WGenerator},           u) where {d} = sum(u) == 1 ?   zero(eltype(u)) : eltype(u)(-Inf)

_cdf(::ArchimedeanCopula{d,IndependentGenerator}, u) where d = prod(u)
_cdf(::ArchimedeanCopula{d,MGenerator},           u) where {d} = minimum(u)
_cdf(::ArchimedeanCopula{d,WGenerator},           u) where {d} = max(1 + sum(u)-d,0)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,IndependentGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    Random.rand!(rng,x)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,MGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,WGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    @assert d==2
    x[1] = rand(rng)
    x[2] = 1-x[1]
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,IndependentGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    Random.rand!(rng,A)
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,MGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    A[1,:] .= rand(rng,size(A,2))
    for i in 2:size(A,1)
        A[i,:] .= A[1,:]
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,WGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    @assert size(A,1) == 2
    A[1,:] .= rand(rng,size(A,2))
    A[2,:] .= 1 .- A[1,:]
    return A
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ClaytonCopula, A::DenseMatrix{<:Real})
    A[:] = inverse_rosenblatt(C, rand(rng, size(A)...))
    return A
end

function rosenblatt(C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}) where {d,TG}
    @assert d == size(u, 1)

    U = zeros(eltype(u), size(u))
    U[1, :] = u[1, :]
    rⱼ = ϕ⁻¹.(C.G, u[1,:])
    rⱼ₋₁ = similar(rⱼ)
    for j in 2:d
        rⱼ₋₁ .= rⱼ
        rⱼ .+= ϕ⁻¹.(C.G, u[j,:]) # so we do not compute too much of them, nor allocate too much.
        U[j, :] .= ϕ⁽ᵏ⁾.(C.G, Val(j - 1), rⱼ) ./ ϕ⁽ᵏ⁾.(C.G, Val(j - 1), rⱼ₋₁)
    end
    return U
end

function rosenblatt(
    C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}
) where {d,TG<:ClaytonGenerator}
    @assert d == size(u, 1)

    U = zeros(eltype(u), size(u))
    U[1, :] = u[1, :]

    for j in 2:d
        U[j, :] .=
            (
                (1 .- j .+ sum(u[1:j, :] .^ (-C.G.θ); dims=1)[:]) ./
                (2 .- j .+ sum(u[1:(j - 1), :] .^ (-C.G.θ); dims=1)[:])
            ) .^ (-1 / C.G.θ - (j - 1))
    end
    return U
end

function inverse_rosenblatt(C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}) where {d,TG}
    @assert d == size(u, 1)
    U = zeros(eltype(u), size(u))
    U[1, :] = u[1, :]
    for i in axes(u, 2)
        Cᵢⱼ = zero(eltype(u))
        for j in 2:d
            Cᵢⱼ += ϕ⁻¹(C.G, U[j - 1, i])
            Dᵢⱼ = ϕ⁽ᵏ⁾(C.G, Val(j - 1), Cᵢⱼ) * u[j,i]
            f(x) = ϕ⁽ᵏ⁾(C.G, Val(j - 1), Cᵢⱼ + ϕ⁻¹(C.G, x)) - Dᵢⱼ
            U[j, i] = Roots.find_zero(f, (eps(1.0), 1.0), Roots.A42())
        end
    end
    return U
end

function inverse_rosenblatt(
    C::ArchimedeanCopula{d,TG}, u::AbstractMatrix{<:Real}
) where {d,TG<:ClaytonGenerator}
    @assert d == size(u, 1)

    U = zeros(eltype(u), size(u))
    U[1, :] = u[1, :]

    for j in 2:d
        U[j, :] =
            (
                1 .+
                (1 .- (j - 1) .+ sum(U[1:(j - 1), :] .^ -C.G.θ; dims=1))[:] .*
                (u[j, :] .^ (-1 / (j - 1 + 1 / C.G.θ)) .- 1)
            ) .^ (-1 / C.G.θ)
    end

    return U
end
