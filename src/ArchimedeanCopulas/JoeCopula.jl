struct JoeCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
# ϕ(  C::Joe,          t) = 1-(1-exp(-t))^(1/C.θ)
# ϕ⁻¹(C::Joe,          t) = -log(1-(1-t)^C.θ)
