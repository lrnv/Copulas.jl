using LinearAlgebra
struct TEVCopula{d, df, MT} <: ExtremeValueCopula{d, MT}
    Σ::MT   # Correlation matrix
    df::Float64 # Degrees of freedom
    function TEVCopula(df::Float64, Σ::MT) where MT
        if size(Σ, 1) != size(Σ, 2)
            throw(ArgumentError("Correlation matrix Σ must be square"))
        end
        if any(eigvals(Σ) .<= 0)
            throw(ArgumentError("Correlation matrix Σ must be positive definite"))
        end
        return new{size(Σ, 1), df, MT}(Σ, df)
    end
end

function construct_R_j(Σ::Matrix{T}, j::Int) where T<:Real

    d = size(Σ, 1)
    if d == 2
        # Devuelve una matriz 1x1 del tipo T con el valor T(1)
        return fill(T(1), 1, 1)
    else
        R_j = Matrix{T}(undef, d - 1, d - 1)
        row_idx = 1
        for row in 1:d
            if row == j
                continue
            end
            col_idx = 1
            for col in 1:d
                if col == j
                    continue
                end
                R_j[row_idx, col_idx] = Σ[row, col]
                col_idx += 1
            end
            row_idx += 1
        end
        return R_j
    end
end

# Función ℓ ajustada para manejar tipos duales
function ℓ(copula::TEVCopula, u::Vector)
    ν = copula.df
    Σ = convert_to_dual(copula.Σ)  # Asegúrate de que Σ maneje tipos duales si es necesario
    d_len = length(u)
    l_u = zero(eltype(Σ))

    for j in 1:d_len
        R_j = construct_R_j(Σ, j)
        terms = Vector{typeof(sqrt(Σ[1,1]))}(undef, d_len - 1)  # Usa el tipo calculado 
        idx = 1
        for i in 1:d_len
            if i != j
                ρ_ij = Σ[i, j]
                term = (sqrt(eltype(Σ)(ν + 1)) / sqrt(eltype(Σ)(1) - ρ_ij^2)) * ((u[i] / u[j])^(-eltype(Σ)(1)/ν) - ρ_ij)
                println("TIPO DE TERMINO:", typeof(term))
                println("valor TERMINO:", term)
                term_value = ForwardDiff.value(term)  # Esto toma sólo el valor real de un Dual
                terms[idx] = term_value  # Ahora está seguro de asignar porque term_value es Float64
                idx += 1
            end
        end

        if d_len == 2
            term_value = ForwardDiff.value(terms[1])  # Convertir solo si es necesario
            T_dist = Distributions.TDist(ν + 1)
            cdf_value = Distributions.cdf(T_dist, term_value)
            l_u += u[j] * cdf_value
        else
            term_values = map(ForwardDiff.value, terms)  # Convertir cada término a Float64
            T_dist = Distributions.MvTDist(ν + 1, R_j)
            l_u += u[j] * Distributions.cdf(T_dist, hcat(term_values...))
        end
    end
    return l_u
end
function convert_to_dual(Σ::Matrix{Float64})
    # Define el tipo Dual apropiado, asociado a Float64 y una etiqueta genérica
    T = ForwardDiff.Dual{Float64}

    # Convertir cada elemento de Σ a Dual
    Σ_dual = map(x -> T(x, 0.0), Σ)
    return Σ_dual
end

# Ejemplo de uso
Σ = [1.0 0.5; 0.5 1.0]
C = TEVCopula(4.0, Σ)
u = [0.4, 0.5]
println("CDF is :", _cdf(C,u))
#_cdf(C,u)
_pdf(C,u)