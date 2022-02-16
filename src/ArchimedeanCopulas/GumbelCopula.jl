struct GumbelCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
GumbelCopula(d,θ) = θ >= 1 ? GumbelCopula{d,typeof(θ)}(θ) : @error "Theta must be greater than 1."
ϕ(  C::GumbelCopula,       t) = exp(-t^(1/C.θ))
ϕ⁻¹(C::GumbelCopula,       t) = (-log(t))^C.θ
τ(C::GumbelCopula) = (C.θ-1)/C.θ
τ⁻¹(::Type{GumbelCopula},τ) =1/(1-τ) 

function radial_dist(C::GumbelCopula)
    α = 1/C.θ
    β = 1
    γ = cos(π/(2C.θ))^C.θ
    δ = C.θ == 1 ? 1 : 0
    AlphaStable([α,β,γ,δ]...) # for the type promotion...
end


# S(α, β, γ , δ) denotes a stable distribution in
# 1-parametrization [16, p. 8] with characteristic exponent α ∈ (0, 2], skewness β ∈ [−1, 1], scale
# γ ∈ [0,∞), and location δ ∈ R