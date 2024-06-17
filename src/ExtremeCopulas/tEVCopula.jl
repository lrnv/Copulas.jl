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
    Σ = copula.Σ
    d_len = length(u)
    l_u = zero(one(eltype(u)))  # Inicializar l_u con el tipo adecuado

    for j in 1:d_len
        R_j = construct_R_j(Σ, j)  # Construir la submatriz R_j
        println("R_j for j=$j: $R_j")  # Imprimir R_j para depuración

        terms = Vector{eltype(u)}()  # Asegurar que el vector terms tenga el tipo correcto
        for i in 1:d_len
            if i != j
                ρ_ij = Σ[i, j]
                term = (sqrt(ν + 1) * one(u[i]) / sqrt(one(u[i]) - ρ_ij^2)) * ((u[i] / u[j])^(-one(u[i])/ν) - ρ_ij * one(u[i]))
                push!(terms, term)
            end
        end
        println("terms for j=$j: $terms")  # Imprimir términos para depuración

        # Convertir temporalmente terms a Float64 para CDF
        terms_float64 = map(x -> ForwardDiff.value(x), terms)
        println("terms_float64 for j=$j: $terms_float64")

        if d_len == 2
            # Para el caso univariado, usar TDist con el parámetro adecuado
            T_dist = Distributions.TDist(ν + 1)
            par = terms_float64[1]
            println("TIPO DE PARAMETRO:", typeof(par))
            l_u += u[j] * Distributions.cdf(T_dist, par)
        else
            T_dist = Distributions.MvTDist(ν + 1, R_j)
            println("TIPOS DE PARAMETROS MULTIVARIADOS:", typeof.(terms_float64))
            l_u += u[j] * Distributions.cdf(T_dist, hcat(terms_float64...))
        end
    end
    return l_u
end

# Ejemplo de uso
Σ = [1.0 0.5; 0.5 1.0]
C = TEVCopula(4.0, Σ)
u = [0.4, 0.5]
println("CDF is :", _cdf(C,u))
#_cdf(C,u)
_pdf(C,u)