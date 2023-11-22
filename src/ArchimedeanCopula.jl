"""
    ArchimedeanCopula{d, TG}

Fields: 
    - G::TG : the generator <: Generator. 

Constructor: 

    ArchimedeanCopula(d::Int,G::Generator)

For some archimedean generator `G::Generator` and some dimenson `d`, this class models the archimedean copula wich has this generator. More formally, a ``d``-monotone archimedean generator is a function ``\\phi`` on ``\\mathbb R_+`` that has these three properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

In the code, this function is implemented as `ϕ(G::Generator,t)`, and its max monotony `d` is given by `max_monotony(G)`. The corresponding d-variate archimedean copula writes: 

```math
C(\\mathbf u) = \\phi^{-1}\\left(\\sum_{i=1}^d \\phi(u_i)\\right)
```

It can be sampled through A Radial simplex decomposition as follows: If ``\\mathbf U \\sim C``, then ``U \\equal R \\mathbf S`` for a random vector ``\\mathbf S \\sim`` `Dirichlet(ones(d))`, that is uniformity on the d-variate simplex, and a non-negative random variable ``R`` that is the Williamson d-transform of `\\phi`. The density, cdf, kendall tau, and may other properties of the Archimedean copula are then directly deduced from this relationship. 

There exists several known parametric generators that are implement in the package. For every `NamedGenerator <: Generator` implemented in the package, we provide a type alias ``NamedCopula{d,...} = ArchimedeanCopula{d,NamedGenerator{...}}` to be able to manipulate the archimedean copulas without too much hassle for known and usefull special cases. 

A generic archimdeanc copula can be constructed as follows: 

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

The following functions have defaults but can be overridden for performance: 

```julia
ϕ⁻¹(C::MyArchimedean, t)        # Inverse of ϕ
ϕ⁽¹⁾(C::MyArchimedean, t)       # first defrivative of ϕ
ϕ⁽ᵈ⁾(C::MyArchimedean,t)        # dth defrivative of ϕ
τ(C::MyArchimedean)             # Kendall tau
τ⁻¹(::Type{MyArchimedean},τ) =  # Inverse kendall tau
fit(::Type{MyArchimedean},data) # fitting.
```

As a bonus, If you know the williamson d-transform of your generator and not your generator itself, you may take a look at [`WilliamsonGenerator`](@ref) that implements them. If you rather know the frailty distribution, take a look at `WilliamsonFromFrailty`.

References: 
    Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct ArchimedeanCopula{d,TG} <: Copula{d}
    G::TG
    function ArchimedeanCopula(d::Int,G::Generator)
        @assert d <= max_monotony(G) "The generator you provided is not d-monotonous according to its max_monotonicity property, and thus this copula does not exists."
        return new{d,typeof(G)}(G)
    end
end
ϕ(C::ArchimedeanCopula{d,TG},t)    where {d,TG} = ϕ(C.G,t)
ϕ⁻¹(C::ArchimedeanCopula{d,TG},t)  where {d,TG} = ϕ⁻¹(C.G,t)
ϕ⁽¹⁾(C::ArchimedeanCopula{d,TG},t) where {d,TG} = ϕ⁽¹⁾(C.G,t)
ϕ⁽ᵏ⁾(C::ArchimedeanCopula{d,TG},k,t) where {d,TG} = ϕ⁽ᵏ⁾(C.G,k,t)
williamson_dist(C::ArchimedeanCopula{d,TG}) where {d,TG} = williamson_dist(C.G,d)


function _cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C,us)
    end
    return ϕ(C,sum_ϕ⁻¹u)
end
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .<= u .<= 1)
        return eltype(u)(-Inf)
    end
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C,ϕ⁻¹u)) # log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    numer = ϕ⁽ᵏ⁾(C, d, sum_ϕ⁻¹u)
    dimension_sign = iseven(d) ? 1.0 : -1.0 #need this for log since (-1.0)ᵈ ϕ⁽ᵈ⁾ ≥ 0.0


    # I am not sure this is the right reasoning :
    if numer == 0
        if sum_logϕ⁽¹⁾ϕ⁻¹u == -Inf
            return Inf
        else
            return -Inf
        end
    else
        return log(dimension_sign*numer) - sum_logϕ⁽¹⁾ϕ⁻¹u
    end
end

# function τ(C::ArchimedeanCopula)  
#     return 4*Distributions.expectation(r -> ϕ(C,r), williamson_dist(C)) - 1
# end

function _archi_rand!(rng,C::ArchimedeanCopula{d},R,x) where d
    # x is assumed to already be random exponentials produced by Random.randexp
    r = rand(rng,R)
    sx = sum(x)
    for i in 1:d
        x[i] = ϕ(C,r * x[i]/sx)
    end
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # By default, we use the williamson sampling. 
    Random.randexp!(rng,x)
    _archi_rand!(rng,C,williamson_dist(C),x)
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # More efficient version that precomputes the williamson transform on each call to sample in batches: 
    Random.randexp!(rng,A)
    R = williamson_dist(C)
    for i in 1:size(A,2)
        _archi_rand!(rng,C,R,view(A,:,i))
    end
    return A
end
function Distributions.fit(::Type{CT},u) where {CT <: ArchimedeanCopula}
    # @info "Archimedean fits are by default through inverse kendall tau."
    d = size(u,1)
    τ = StatsBase.corkendall(u')
    # Then the off-diagonal elements of the matrix should be averaged: 
    avgτ = (sum(τ) .- d) / (d^2-d)
    θ = τ⁻¹(CT,avgτ)
    return CT(d,θ)
end

τ(C::ArchimedeanCopula{d,TG}) where {d,TG} = τ(C.G)
function τ⁻¹(::Type{T},τ_val) where {T<:ArchimedeanCopula}
    return τ⁻¹(generatorof(T),τ_val)
end
ρ(C::ArchimedeanCopula{d,TG}) where {d,TG} = ρ(C.G)
function ρ⁻¹(::Type{T},ρ_val) where {T<:ArchimedeanCopula}
    return ρ⁻¹(generatorof(T),ρ_val)
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
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,IndependentGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    Random.rand!(rng,A)
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,MGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    A[1,:] .= rand(rng,size(A,2))
    for i in 2:size(A,1)
        A[i,:] .= A[1,:]
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,WGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    @assert size(A,1) == 2
    A[1,:] .= rand(rng,size(A,2))
    A[2,:] .= 1 .- A[1,:]
    return A
end

