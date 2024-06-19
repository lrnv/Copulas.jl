struct TEVCopula{d, df, MT} <: ExtremeValueCopula{d, MT}
    Σ::MT   # Correlation matrix
    function TEVCopula(df::Int64, Σ::MT) where MT
        if size(Σ, 1) != size(Σ, 2)
            throw(ArgumentError("Correlation matrix Σ must be square"))
        end
        if !LinearAlgebra.isposdef(Σ)
            throw(ArgumentError("Correlation matrix Σ must be positive definite"))
        end
        return new{size(Σ, 1), df, MT}(Σ)
    end
end

function construct_R_j(Σ::Matrix{T}, j::Int) where T<:Real
    d = size(Σ, 1)
    R_j = Matrix{T}(undef, d - 1, d - 1)
    if d == 2
        R_j .= 1
        return R_j
    else
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
function ℓ(C::TEVCopula{2, ν, MT}, u) where {ν,MT}
    D  = Distributions.TDist(ν + 1)
    a1 = sqrt(1 - C.Σ[1, 2]^2)
    a2 = sqrt(1 - C.Σ[2, 1]^2)
    r  = sqrt(ν+1)
    s  = (u[1] / u[2])^(-1/ν)
    v1 = (r / a1) * (s   - C.Σ[1, 2])
    v2 = (r / a2) * (1/s - C.Σ[2, 1])
    return u[2] * Distributions.cdf(D, v1) +  u[1] * Distributions.cdf(D, v2)
end
function ℓ(C::TEVCopula{d, ν, MT}, u) where {d,ν,MT}
    l_u = zero(eltype(u))  # Inicializar l_u con el tipo adecuado
    Ds = (Distributions.MvTDist(ν + 1, construct_R_j(C.Σ, j)) for j in 1:d)
    for j in 1:d
        terms = zero(u)
        for i in 1:d
            terms[i] = (sqrt(ν + 1) / sqrt(1 - C.Σ[i, j]^2)) * ((u[i] / u[j])^(-1/ν) - C.Σ[i, j])
        end
        deleteat!(terms,j) # remove the zero at the jth term. 
        l_u += u[j] * Distributions.cdf(Ds[j], terms)
    end
    return l_u
end