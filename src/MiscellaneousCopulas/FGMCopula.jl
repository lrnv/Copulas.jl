"""
    FGMCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

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

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
* [blier2022stochastic](@cite) Blier-Wong, C., Cossette, H., & Marceau, E. (2022). Stochastic representation of FGM copulas using multivariate Bernoulli random variables. Computational Statistics & Data Analysis, 173, 107506.
"""
struct FGMCopula{d, Tθ, Tf} <: Copula{d}
    θ::Tθ
    fᵢ::Tf
    function FGMCopula(d, θ)
        vθ = θ isa Vector ? promote(θ...,1.0)[1:end-1] : [promote(θ,1.0)[1]]
        if  all(θ .== 0)
            return IndependentCopula(d)
        end
        # Check first restrictions on parameters
        any(abs.(vθ) .> 1) && throw(ArgumentError("Each component of the parameter vector must satisfy that |θᵢ| ≤ 1"))
        length(vθ) != 2^d - d - 1 && throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
        
        # Last check: 
        for epsilon in Base.product(fill([-1, 1], d)...)
            if 1 + _fgm_red(vθ, epsilon) < 0
                throw(ArgumentError("Invalid parameters. The parameters do not meet the condition to be an FGM copula"))
            end
        end
        
        # Now construct the stochastic representation:
        wᵢ = [_fgm_red(vθ, 1 .- 2*Base.reverse(digits(i, base=2, pad=d))) for i in 0:(2^d-1)]
        fᵢ = Distributions.DiscreteNonParametric(0:(2^d-1), (1 .+ wᵢ)/2^d)
        return new{d, typeof(vθ), typeof(fᵢ)}(vθ, fᵢ)
    end
    FGMCopula{D, T1, T2}(d, θ) where {D, T1, T2} = FGMCopula(d, θ)
end
Base.eltype(C::FGMCopula) = eltype(C.θ)

# Fitting/params interface
Distributions.params(C::FGMCopula) = (θ = collect(C.θ),)
_example(::Type{<:FGMCopula}, d) = FGMCopula(d, fill(0.1, 2^d - d - 1))
_available_fitting_methods(::Type{<:FGMCopula{2}}) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(::Type{<:FGMCopula}) = (:mle,)

# Compute the maximal λ so that all FGM constraints are strictly satisfied
function _max_lambda(β, d)
    λmax = 1.0
    for epsilon in Base.product(fill([-1, 1], d)...) 
        red = _fgm_red(β, epsilon)
        if red != 0
            λmax = min(λmax, 1 / abs(red))
        end
    end
    # Also ensure |θᵢ| < 1 for all i
    for b in β
        if b != 0
            λmax = min(λmax, 1 / abs(b))
        end
    end
    # Stay strictly inside the polytope
    return 0.999 * λmax
end

function _rebound_params(::Type{<:FGMCopula}, d, α)
    if d == 2
        # Only one parameter, strictly invertible
        return (; θ = tanh.(α))
    end
    # For d >= 3, use a safe directional mapping (not fully surjective, but stays in the interior)
    β = α
    normβ = LinearAlgebra.norm(β)
    if normβ == 0
        θ = zeros(length(β))
    else
        direction = β / normβ
        # Find the maximal λ in this direction, then stay well inside
        λmax = _max_lambda(direction, d)
        r = exp(normβ) / (1 + exp(normβ))
        λ = 0.95 * λmax * r # 0.95 to stay strictly inside
        θ = λ * direction
    end
    return (; θ = θ)
end

function _unbound_params(::Type{<:FGMCopula}, d, θ)
    θvec = collect(θ.θ)
    if d == 2
        # Only one parameter, strictly invertible
        return atanh.(θvec)
    end
    # For d >= 3, use the fast directional mapping, but ensure safety
    normθ = LinearAlgebra.norm(θvec)
    if normθ == 0
        return zeros(length(θvec))
    end
    direction = θvec / normθ
    λmax = _max_lambda(direction, d)
    # Clamp r to (0, 1-eps()) to avoid Inf/NaN
    r = clamp(normθ / λmax, 0.0, 1.0 - eps())
    normβ = log(r / (1 - r))
    return direction * normβ
end
function _fgm_red(θ, v)
    # This function implements the reduction over combinations of the fgm copula. 
    # It is non-alocative thus performant :)
    rez, d, i = zero(eltype(v)), length(v), 1
    for k in 2:d
        for indices in Combinatorics.combinations(1:d, k)
            rez += θ[i] * prod(v[indices])
            i = i+1
        end
    end
    return rez
end
_cdf(fgm::FGMCopula, u::Vector{T}) where {T} = prod(u) * (1 + _fgm_red(fgm.θ, 1 .-u))
Distributions._logpdf(fgm::FGMCopula, u) = log1p(_fgm_red(fgm.θ, 1 .-2u))
function Distributions._rand!(rng::Distributions.AbstractRNG, fgm::FGMCopula{d, Tθ, Tf}, x::AbstractVector{T}) where {d,Tθ, Tf, T <: Real}
    I = Base.reverse(digits(rand(rng,fgm.fᵢ), base=2, pad=d))
    V₀ = rand(rng, d)
    V₁ = rand(rng, d)
    x .= 1 .- sqrt.(V₀) .* (V₁ .^ I)
    return x
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

# Subsetting colocated
function SubsetCopula(C::FGMCopula{d,Tθ,Tf}, dims::NTuple{p, Int}) where {d,Tθ,Tf,p}
    if p==2
        i = 1
        for indices in Combinatorics.combinations(1:d, 2)
            all(indices .∈ Ref(dims)) && return FGMCopula(2,C.θ[i])
            i = i+1
        end
        @error("Somethings wrong...")
    end
    # Build mapping to gather θ' in the canonical order for dimension p
    combos_by_k = [collect(Combinatorics.combinations(1:d, k)) for k in 2:d]
    offs = Vector{Int}(undef, d)
    offs[1] = 0  # unused for k=1
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

DistortionFromCop(C::FGMCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) = BivFGMDistortion(float(C.θ[1]), Int8(js[1]), float(uⱼₛ[1]))

