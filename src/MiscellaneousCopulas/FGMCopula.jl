"""
    FGMCopula{d}(θ)
    FGMCopula(d, θ)

The multivariate Farlie–Gumbel–Morgenstern (FGM) copula of dimension d has ``2^d-d-1`` parameters ``\\theta`` and

```math
C(\\boldsymbol{u})=\\prod_{i=1}^{d}u_i \\left[1+ \\sum_{k=2}^{d}\\sum_{1 \\leq j_1 < \\cdots < j_k \\leq d} \\theta_{j_1 \\cdots j_k} \\bar{u}_{j_1}\\cdots \\bar{u}_{j_k} \\right],
```

where ``\\bar{u} = 1 - u``.

Special cases:
- When d=2 and θ = 0, it is the IndependentCopula.

More details about Farlie-Gumbel-Morgenstern (FGM) copula are found in [nelsen2006](@cite).
We use the stochastic representation from [blier2022stochastic](@cite) to obtain random samples.

The parameter vector is ordered by interaction order and then by coordinate
combination. Componentwise bounds `|θᵢ| ≤ 1` are necessary but not sufficient;
the constructor also checks all hypercube-corner inequalities. Parameter count
grows exponentially with `d`, while attainable dependence remains relatively
weak. In dimension two, the admissible endpoints `θ = ±1` remain ordinary FGM
copulas; the family does not attain either Fréchet--Hoeffding bound.

See also: [`Copula`](@ref), [`IndependentCopula`](@ref), [`τ`](@ref),
[`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
* [blier2022stochastic](@cite) Blier-Wong, C., Cossette, H., & Marceau, E. (2022). Stochastic representation of FGM copulas using multivariate Bernoulli random variables. Computational Statistics & Data Analysis, 173, 107506.
"""
struct FGMCopula{d, Tθ, Tf} <: Copula{d}
    θ::Tθ
    fᵢ::Tf
    function FGMCopula{d}(vθ::Vector) where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
        any(abs.(vθ) .> 1) && throw(ArgumentError("Each component of the parameter vector must satisfy that |θᵢ| ≤ 1"))
        length(vθ) != 2^d - d - 1 && throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))

        for epsilon in Base.product(fill([-1, 1], d)...)
            if 1 + _fgm_red(vθ, epsilon) < 0
                throw(ArgumentError("Invalid parameters θ = $vθ. The parameters do not meet the condition to be an FGM copula"))
            end
        end

        wᵢ = [_fgm_red(vθ, 1 .- 2*Base.reverse(digits(i, base=2, pad=d))) for i in 0:(2^d-1)]
        support = 0:(2^d-1)
        probabilities = (1 .+ wᵢ) / 2^d
        Tf = Distributions.DiscreteNonParametric{
            eltype(support), eltype(probabilities), typeof(support), typeof(probabilities),
        }
        fᵢ = Distributions.DiscreteNonParametric(support, probabilities)::Tf
        return new{d, typeof(vθ), typeof(fᵢ)}(vθ, fᵢ)
    end
end
FGMCopula{d}(θ::Real) where {d} = FGMCopula{d}([float(θ)])
FGMCopula{d}(θ::Tuple) where {d} = FGMCopula{d}(collect(float.(θ)))
FGMCopula{d}(θ::AbstractVector) where {d} = FGMCopula{d}(collect(float.(θ)))
FGMCopula(d, θ) = FGMCopula{d}(θ)
(::Type{<:FGMCopula{D,Tθ,Tf}})(d::Int, θ) where {D,Tθ,Tf} = FGMCopula{d}(θ)
function _fgm_red(θ, v)
    rez, d, i = zero(promote_type(eltype(θ), eltype(v))), length(v), 1
    for k in 2:d
        for indices in Combinatorics.combinations(1:d, k)
            rez += θ[i] * prod(v[indices])
            i = i+1
        end
    end
    return rez
end
Base.eltype(C::FGMCopula) = eltype(C.θ)

Distributions.params(C::FGMCopula) = (collect(C.θ),)
_available_fitting_methods(::Type{<:FGMCopula}, d) = d==2 ? (:mle, :itau, :irho, :ibeta) : (:mle,)

# The bivariate FGM domain is an ordinary bounded scalar chart. Higher-dimensional
# FGM parameters satisfy coupled hypercube-corner inequalities and deliberately
# keep their specialized constrained optimizer below.
function Paramorph.param_space(::Type{<:FGMCopula}, d::Integer)
    d == 2 || throw(ArgumentError(
        "multivariate FGM has coupled parameter constraints not represented by a Paramorph product space"))
    return Paramorph.Bounded(:θ, -1.0, 1.0)
end

function _cdf(fgm::FGMCopula{d}, u::Vector{T}) where {d,T}
    return prod(u) * (1 + _fgm_red(fgm.θ, 1 .-u))
end
copula_measure_style(::FGMCopula) = AbsolutelyContinuousMeasure()
Distributions._logpdf(fgm::FGMCopula, u) = log1p(_fgm_red(fgm.θ, 1 .-2u))
function Distributions._rand!(rng::Distributions.AbstractRNG, fgm::FGMCopula{d, Tθ, Tf}, A::AbstractMatrix{T}) where {d,Tθ, Tf, T <: Real}
    size(A, 1) == d || throw(DimensionMismatch("output matrix must have $d rows"))
    Random.rand!(rng, A)
    V₁ = rand(rng, T, size(A))
    states = rand(rng, fgm.fᵢ, size(A, 2))
    @inbounds for (j, col) in enumerate(axes(A, 2)), (i, row) in enumerate(axes(A, 1))
        bit = (states[j] >> (d - i)) & 1
        A[row, col] = one(T) - sqrt(A[row, col]) * (iszero(bit) ? one(T) : V₁[i, j])
    end
    return A
end
τ(fgm::FGMCopula{2, Tθ, Tf}) where {Tθ,Tf} = (2*fgm.θ[1])/9
function τ⁻¹(::Type{<:FGMCopula}, τ)
    if !all(-2/9 <= τi <= 2/9 for τi in τ)
        throw(ArgumentError("For the FGM copula, tau must be in [-2/9, 2/9]."))
    end
    return max.(min.(9 * τ / 2, 1), -1)
end
ρ(fgm::FGMCopula{2, Tθ, Tf}) where {Tθ,Tf} = fgm.θ[1]/3
function ρ⁻¹(::Type{<:FGMCopula}, ρ)
    if !all(-1/3 <= ρi <= 1/3 for ρi in ρ)
        throw(ArgumentError("For the FGM copula, rho must be in [-1/3, 1/3]."))
    end
    return max.(min.(3 * ρ, 1), -1)
end

function SubsetCopula(C::FGMCopula{d,Tθ,Tf}, dims::NTuple{p, Int}) where {d,Tθ,Tf,p}
    if p==2
        i = 1
        for indices in Combinatorics.combinations(1:d, 2)
            all(indices .∈ Ref(dims)) && return FGMCopula(2,C.θ[i])
            i = i+1
        end
        @error("Somethings wrong...")
    end
    combos_by_k = [collect(Combinatorics.combinations(1:d, k)) for k in 2:d]
    offs = Vector{Int}(undef, d)
    offs[1] = 0
    acc = 0
    for k in 2:d
        offs[k] = acc
        acc += length(combos_by_k[k-2+1])
    end
    θ′ = Vector{eltype(C.θ)}()
    for k in 2:p
        for pos_combo in Combinatorics.combinations(1:p, k)
            orig_combo = Tuple(dims[i] for i in pos_combo)
            list_k = combos_by_k[k-2+1]
            idx_in_k = findfirst(==(orig_combo), list_k)
            @assert idx_in_k !== nothing
            push!(θ′, C.θ[offs[k] + idx_in_k])
        end
    end
    return FGMCopula(p, θ′)
end

distortion(C::FGMCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) = BivFGMDistortion(float(C.θ[1]), Int8(js[1]), float(uⱼₛ[1]))

function _fit(
    CT::Type{<:FGMCopula}, U,
    method::Union{Val{:itau},Val{:irho},Val{:ibeta}};
    weights=nothing,
)
    size(U, 1) == 2 || throw(ArgumentError("rank fitting for FGM is available only in dimension two"))
    return _fit(CT, U, Val(2), method; weights)
end

function _fit(CT::Type{<:FGMCopula}, U, ::Val{:mle}; weights=nothing)
    d = size(U,1)

    if d == 2
        return _fit(CT, U, Val(2), Val(:mle); weights)
    end

    cop(θ) = FGMCopula(d, θ)
    nparams = 2^d - d - 1
    θ₀ = fill(0.5 / nparams, nparams)

    function barrier_penalty(θ; μ=1e-3, soft=false)
        total = 0.0
        for ε in Base.product(fill([-1,1], d)...)
            v = 1 + _fgm_red(θ, ε)
            if soft
                total += LogExpFunctions.log1pexp(-10v) / 10
            else
                v <= 0 && return Inf
                total -= μ * log(v)
            end
        end
        return μ * total
    end

    function loss(θ)
        try
            C = cop(θ)
            return -_weighted_loglikelihood(C, U, weights) + barrier_penalty(θ)
        catch
            return 1e10
        end
    end

    res = Optim.optimize(loss, θ₀, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())
    θhat = Optim.minimizer(res)
    return FGMCopula(d, θhat)
end
