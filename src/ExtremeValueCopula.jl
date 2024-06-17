abstract type ExtremeValueCopula{d, TG} <: Copula{d} end

# Define la función ℓ como una función placeholder si no está definida
function ℓ(C::ExtremeValueCopula, t::Vector)
    throw(ArgumentError("Function ℓ must be defined for specific copula"))
end

# Función para calcular derivadas mixtas de ℓ
function D_B_ℓ(C::ExtremeValueCopula{d, TG}, t::Vector{<:Real}, B::Vector{Int}) where {d, TG}
    f = x -> ℓ(C, x)
    
    if length(B) == 1
        return ForwardDiff.gradient(f, t)[B[1]]
    elseif length(B) == 2
        return ForwardDiff.hessian(f, t)[B[1], B[2]]
    else
        # Para derivadas de orden mayor a 2, calculamos la Hessiana y luego derivamos nuevamente.
        hessian_f = ForwardDiff.hessian(f, t)
        
        # Cache para almacenar los resultados intermedios de las derivadas
        cache = Dict{Tuple{Int, Int}, Matrix{<:Real}}()
        cache[(1, 2)] = hessian_f
        
        for i in 3:length(B)
            key = (B[i-2], B[i-1])
            if !haskey(cache, key)
                cache[key] = ForwardDiff.gradient(x -> cache[(B[i-3], B[i-2])][B[i-2], B[i-1]], t)
            end
            hessian_f = cache[key]
        end
        return hessian_f[B[end-1], B[end]]
    end
end

# Función para generar todas las particiones de un conjunto de longitud d en m bloques
function partitions_of_length(d::Int, m::Int)
    iter = partitions(collect(1:d))
    return [p for p in iter if length(p) == m]
end

# Función genérica para la CDF que utiliza la función de cola estable ℓ
function _cdf(C::ExtremeValueCopula, u::AbstractArray)
    t = -log.(u)
    return ForwardDiff.value(exp(-ℓ(C, t)))
end

## Función genérica para la PDF que utiliza la función de cola estable ℓ y sus derivadas mixtas
function _pdf(C::ExtremeValueCopula, u::AbstractArray)
    t = -log.(u)
    CDF_value = exp(-ℓ(C, t))
    Independence_copula = prod(u)  # La cópula de independencia es simplemente el producto de u_i
    
    sum_terms = 0.0
    d = length(u)
    for m in 1:d
        sign = (-1)^(d - m)
        partitions = partitions_of_length(d, m)
        for π in partitions
            prod_term = 1.0
            for B in π
                prod_term *= D_B_ℓ(C, t, B)
            end
            sum_terms += sign * prod_term
        end
    end
    
    return (CDF_value / Independence_copula) * sum_terms
end