"""
    ExtremeValueCopula{P}

This empty docstring needs to be filled. 


References:
* Add a few references
"""
abstract type ExtremeValueCopula{P} <: Copula{2} end

# Función genérica para A
function A(C::ExtremeValueCopula, t::Real)
    throw(ArgumentError("Function A must be defined for specific copula"))
end

function dA(C::ExtremeValueCopula, t::Real)
    ForwardDiff.derivative(t -> A(C, t), t)
end

function d²A(C::ExtremeValueCopula, t::Real)
    ForwardDiff.derivative(t -> dA(C, t), t)
end

function ℓ(C::ExtremeValueCopula, t::Vector)
    sumu = sum(t)
    vectw = t[1] / sumu
    return sumu * 𝘈(C, vectw)
end

# Función CDF para ExtremeValueCopula
function _cdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = abs.(log.(u)) # 0 <= u <= 1 so abs == neg, but return corectly 0 instead of -0 when u = 1. 
    return exp(-ℓ(C, t))
end

# Función genérica para calcular derivadas parciales de ℓ
function D_B_ℓ(C::ExtremeValueCopula, t::Vector{Float64}, B::Vector{Int})
    f = x -> ℓ(C, x)

    if length(B) == 1
        return ForwardDiff.gradient(f, t)[B[1]]
    elseif length(B) == 2
        return ForwardDiff.hessian(f, t)[B[1], B[2]]
    else
        throw(ArgumentError("Higher order partial derivatives are not required for bivariate case"))
    end
end

# Función PDF para ExtremeValueCopula usando ℓ
function _pdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return c * (-D12 + D1 * D2) / (u[1] * u[2])
end
function Distributions._logpdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return log(c) + log(-D12 + D1 * D2) - log(u[1] * u[2])
end
# Definir la función para calcular τ
function τ(C::ExtremeValueCopula)
    integrand(x) = begin
        a = A(C, x)
        da = dA(C, x)
        return (x * (1 - x) / a) * da
    end
    
    integrate, _ = QuadGK.quadgk(integrand, 0.0, 1.0)
    return integrate
end

function ρₛ(C::ExtremeValueCopula)
    integrand(x) = 1 / (1 + A(C, x))^2
    
    integral, _ = QuadGK.quadgk(integrand, 0, 1)
    
    ρs = 12 * integral - 3
    return ρs
end
# Función para calcular el coeficiente de dependencia en el límite superior
function λᵤ(C::ExtremeValueCopula)
    return 2(1 - A(C, 0.5))
end

function λₗ(C::ExtremeValueCopula)
    if A(C, 0.5) > 0.5
        return 0
    else
        return 1
    end
end

function probability_z(C::ExtremeValueCopula, z)
    num = z*(1 - z)*d²A(C, z)
    dem = A(C, z)*_pdf(ExtremeDist(C), z)
    p = num / dem
    return clamp(p, 0.0, 1.0)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula, x::AbstractVector{T}) where {T<:Real}
    u1, u2 = rand(rng, Distributions.Uniform(0,1), 2)
    z = rand(rng, ExtremeDist(C))
    p = probability_z(C, z)
    if p < -eps() || p > eps()
        p = 0
    end
    c = rand(rng, Distributions.Bernoulli(p))
    w = 0
    if c == 1
        w = u1
    else
        w = u1*u2
    end
    a = A(C, z)
    x[1] = w^(z/a)
    x[2] = w^((1-z)/a)
    return x
end