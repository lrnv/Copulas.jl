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
struct FGMCopula{d, T} <: Copula{d}
    θ::Vector{T}
    function FGMCopula(d, θ)
        if !(typeof(θ)<:Vector)
            vθ = [θ]
        else
            vθ = θ
        end
        if d < 2
            throw(ArgumentError("Dimension (d) must be greater than or equal to 2"))
        end
        if  all(θ .== 0)
            return IndependentCopula(d)
        end
        if any(abs.(vθ) .> 1)
            throw(ArgumentError("each component of the parameter vector must satisfy that |θᵢ| ≤ 1"))
        end
        if length(vθ) != 2^d - d - 1
            throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
        end
        # Verificar las restricciones en los parámetros
        for epsilon in collect(Base.product(fill([-1, 1], d)...))
            # Llamar a la función para generar el vector de coeficientes
            coefficients_vector = Vector{Int}() ################################################## <<<<<------ Please change this
    
            for k in 2:d
                for indices in Combinatorics.combinations(1:d, k)
                    product = prod(epsilon[indices])
                    push!(coefficients_vector, product)
                end
            end
            # Calcular la combinación lineal
            resultado = 1+sum(vθ .* coefficients_vector)
            if resultado < 0
                throw(ArgumentError("Invalid parameters. The parameters do not meet the condition to be an FGM copula"))
            end
        end
        return new{d, eltype(vθ)}(vθ)
    end 
end
Base.eltype(::FGMCopula{d,T}) where {d,T} = T
function func_aux(vectors,d)
    # Helper function to calculate all possible combinations of uᵢ
    products = Vector{eltype(vectors)}() ###################################################################### <<<<<------ Please change this
    # Iterate over all possible combinations of k elements, for k = 2, 3, ..., d
    for k in 2:d
      # Iterate over all possible combinations of k elements in u
      for indices in Combinatorics.combinations(1:d, k)
        #Calculate the product of the u values in the combination
        product = prod(vectors[indices])
        # Add the product to the product vector
        push!(products, product)
      end
    end
    return products
end
function _cdf(fgm::FGMCopula, u::Vector{T}) where {T}
    d = length(fgm)
    term1 = prod(u)
    term2 = sum(fgm.θ .* func_aux(1 .-u,d))
    return term1 * (1+term2)
end
function Distributions._logpdf(fgm::FGMCopula, u::Vector{T}) where {T}
    d = length(fgm)
    term = sum(fgm.θ .* func_aux(1 .- 2u,d))
    return log1p(term)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, fgm::FC, x::AbstractVector{T}) where {FC <: FGMCopula, T <: Real}
    d = length(fgm)
    θ = fgm.θ
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
        # Itera sobre cada dimensión y decide el valor usando get_pmf
        I = zeros(d)
        for i in 1:d
            if length(θ) != 2^d - d - 1
                throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
            end
            # Asegurar que i sea un vector de enteros
            signs = Vector{T}() ################################################################### <<<<<------ Please change this
            # Iterar sobre todas las posibles combinaciones de k elementos, para k = 2, 3, ..., dim
            for k in 2:d
                # Iterar sobre todas las posibles combinaciones de k elementos en índices
                for indices in Combinatorics.combinations(1:d, k)
                    # Calcular la suma de los valores de los vectores en la combinación
                    suma = sum(I[indices])
                    # Agregar el signo al vector de signos
                    push!(signs, (-1)^suma)
                end
            end
            term = sum(signs .* θ)
            I[i] = rand(rng) < (1 / 2^d) * (1 + term)
        end
        V0 = rand(d)
        V1 = rand(d)
        for j in 1:d
            U_j = 1-sqrt(1-V0[j])*(1-V1[j])^(I[j])
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