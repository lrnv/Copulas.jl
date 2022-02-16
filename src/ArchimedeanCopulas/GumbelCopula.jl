struct GumbelCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
# ϕ(  C::Gumbel,       t) = exp(-t^(1/C.θ))
# ϕ⁻¹(C::Gumbel,       t) = (-log(t))^C.θ
