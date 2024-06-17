
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
function ℓ(G::GalambosCopula, t::Vector)
    θ = G.θ
    d = length(t)
    result = 0.0
    for j in 1:d
        sign = (-1)^(j + 1)
        for subset in combinations(1:d, j)
            inner_sum = 0.0
            for k in subset
                inner_sum += t[k]^(-θ)
            end
            result += sign * inner_sum^(-1/θ)
        end
    end
    return result
end
G = GalambosCopula(2,2.5)
u = [0.5, 0.6]

_cdf(G,u)
_pdf(G,u)

