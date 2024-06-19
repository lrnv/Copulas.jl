struct GalambosCopula{d, P} <: ExtremeValueCopula{d, P}
    θ::P  # Copula parameter
    function GalambosCopula(d, θ)
        if θ <= 0
            throw(ArgumentError("Theta must be > 0"))
        else
            return new{d, typeof(θ)}(θ)
        end
    end
end

# Función para la dependencia de cola estable ℓ para la cópula de Galambos
cmb(d,j) = Combinatorics.combinations(1:d, j)
function ℓ(G::GalambosCopula{d,P}, t) where {d,P}
    tpθ = t .^ G.θ
    return sum((-1)^(j+1) * sum(sum(tpθ[s])^(-1/G.θ)  for s in cmb(d,j)) for j in 1:d)
end
