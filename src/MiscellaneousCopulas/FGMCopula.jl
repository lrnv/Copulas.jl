"""
FGMCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    FGMCopula(d, θ)

The Multivariate Farlie-Gumbel-Morgenstern (FGM) copula of dimension d has ``2^d-d-1`` parameters ``\\theta`` and function

```math
C(\\boldsymbol{u})=\\prod_{i=1}^{d}u_i \\left[1+ \\sum_{k=2}^{d}\\sum_{1 \\leq j_1 < \\cdots < j_k \\leq d} \\theta_{j_1 \\cdots j_k} \\bar{u}_{j_1}\\cdots \\bar{u}_{j_k} \\right],
```

where `` \\bar{u}=1-u``.

More details about Farlie-Gumbel-Morgenstern (FGM) copula are found in :
    
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.38.

We use the stochastic representation of the copula to obtain random samples.
    
    Blier-Wong, C., Cossette, H., & Marceau, E. (2022). Stochastic representation of FGM copulas using multivariate Bernoulli random variables. Computational Statistics & Data Analysis, 173, 107506.

It has a few special cases:
- When d=2 and θ = 0, it is the IndependentCopula.
"""
#import Base.Iterators: product
import Base.Iterators: product
struct FGMCopula{d, T} <: Copula{d}
    θ::Vector{Float64}

    function FGMCopula(d::Int, θ::Vector{Float64}) where {T}
        if d < 2
            throw(ArgumentError("Dimension (d) must be greater than or equal to 2"))
        end
        if  all(θ .== 0)
            return IndependentCopula(d)
        end
        
        if any(abs.(θ) .> 1)
            throw(ArgumentError("each component of the parameter vector must satisfy that |θᵢ| ≤ 1"))
        end
        
        expected_params_length = 2^d - d - 1
        if length(θ) != expected_params_length
            throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
        end
        
        # Verificar las restricciones en los parámetros
        verify_parameters(d, θ)
        
        return new{d, Vector{Float64}}(θ)

    end 
    function FGMCopula(d::Int, θ::T) where {T}
        return FGMCopula(d, [θ])
    end
    function verify_parameters(d, θ)
        combinaciones_epsilons = collect(Base.product(fill([-1, 1], d)...))
        
            for epsilon in combinaciones_epsilons
                # Llamar a la función para generar el vector de coeficientes
                coefficients_vector = generate_coefficients(epsilon, d)
        
                # Calcular la combinación lineal
                resultado = 1+LinearAlgebra.dot( θ, coefficients_vector)
          if resultado < 0
                    throw(ArgumentError("Invalid parameters. The parameters do not meet the condition to be an FGM copula"))
                end
    end
      end
    function generate_coefficients(epsilon, d)
        coefficients = Vector{Int}()
    
        for k in 2:d
            for indices in combinations(1:d, k)
                product = prod(epsilon[indices])
                push!(coefficients, product)
            end
        end
    
        return coefficients
    end   
end

Base.length(fgm::FGMCopula{d,T}) where {d,T} = d
Base.eltype(fgm::FGMCopula{d,T}) where {d,T} = Base.eltype(fgm.θ)
# Helper function to calculate all possible combinations of uᵢ
function func_aux(vectors,dim)
    products = Vector{Float64}()
    # Iterate over all possible combinations of k elements, for k = 2, 3, ..., d
    for k in 2:dim
      # Iterate over all possible combinations of k elements in u
      for indices in combinations(1:dim, k)
        #Calculate the product of the u values in the combination
        product = prod(vectors[indices])
        # Add the product to the product vector
        push!(products, product)
      end
    end
    return products
end
# CDF calculation for F-G-M Copula
function Distributions.cdf(fgm::FGMCopula, u::Vector{T}) where {T}
    d = length(fgm)
    if length(u) != d
        throw(ArgumentError("Dimension mismatch between copula and input vector"))
    end
    v = 1 .-u
    term1 = prod(u)
    println("theta", fgm.θ )
    term2 = LinearAlgebra.dot(fgm.θ, func_aux(v,d))
    return term1 * (1+term2)
end
# PDF calculation for F-G-M Copula
function Distributions.pdf(fgm::FGMCopula, u::Vector{T}) where {T}
    d = length(fgm)
    if length(u) != d
        throw(ArgumentError("Dimension mismatch between copula and input vector"))
    end
    v = 1 .- 2*u
    term = LinearAlgebra.dot(fgm.θ, func_aux(v,d))
    
    return 1+term
end
# stochastic representation 
struct fgmDistribution{T<:Real} <: Distributions.DiscreteMultivariateDistribution
    θ::Vector{T}
    d::Int
    function fgmDistribution(θ::Vector{T}, d::Int) where {T <: Real}
        if length(θ) != 2^d - d - 1
            throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
        end
        new{T}(θ, d)
    end
end
Base.length(F::fgmDistribution{T}) where {T} = F.d
function Distributions._rand!(rng::Distributions.AbstractRNG, distribution::fgmDistribution, x::AbstractVector{T}) where {T <: Real}
    dim = distribution.d
    
    # Itera sobre cada dimensión y decide el valor usando get_pmf
    for i in 1:dim
        x[i] = rand(rng) < stochastic_fgm(distribution.θ, x)
    end
    
    return x
end

function stochastic_fgm(theta, i)
    dim = length(i)
    if length(theta) != 2^dim - dim - 1
        throw(ArgumentError("Number of parameters (θ) must match the dimension ($dim): 2ᵈ-d-1"))
    end

    # Asegurar que i sea un vector de enteros
    signs = Vector{Float64}()
    
    # Iterar sobre todas las posibles combinaciones de k elementos, para k = 2, 3, ..., dim
    for k in 2:dim
        # Iterar sobre todas las posibles combinaciones de k elementos en índices
        for indices in combinations(1:dim, k)
            # Calcular la suma de los valores de los vectores en la combinación
            suma = sum(i[indices])
            # Agregar el signo al vector de signos
            push!(signs, (-1)^suma)
        end
    end

    term = LinearAlgebra.dot(signs, theta)
    return (1 / 2^dim) * (1 + term)
end
#########

# Sampling Copula FGM
function Distributions._rand!(rng::Distributions.AbstractRNG, fgm::FC, x::AbstractVector{T}) where {FC <: FGMCopula, T <: Real}
    d = length(fgm)
    if d == 2
        u = rand(rng)
        t = rand(rng)
        a = 1.0 .+ fgm.θ .* (1.0-2.0*u)
        b = sqrt.(a.^2 .-4.0 .*(a .-1.0).*t)
        v = (2.0 .*t) ./(b .+ a)
        x[1] = u
        x[2] = v[1]
        return x
    elseif d > 2
        I = rand(fgmDistribution(fgm.θ,d), 1)
        V0 = rand(d)
        V1 = rand(d)
            for j in 1:length(fgm)
                U_j = 1.0-sqrt(1.0-V0[j])*(1.0-V1[j])^(I[j])
                x[j] = U_j
            end
        return x
    end
end
τ(fgm::FGMCopula) = (2*fgm.θ[1])/9
function τ⁻¹(::Type{FGMCopula}, τ)
    if any(τ .< -2/9 .|| τ .> 2/9)
        println("For the FGM copula, tau must be in [-2/9, 2/9].")
    end
    return max.(min.(9 * τ / 2, 1), -1)
end
ρ(fgm::FGMCopula) = (1*fgm.θ)/3
function ρ⁻¹(::Type{FGMCopula}, ρ)
    if any(ρ .< -1/3 .|| ρ .> 1/9)
        println("For the FGM copula, rho must be in [-1/3, 1/3].")
    end
    return max.(min.(3 * ρ, 1), -1)
end
