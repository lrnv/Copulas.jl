struct JoeCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
JoeCopula(d,θ) = θ >= 1 ? JoeCopula{d,typeof(θ)}(θ) : @error "Theta must be greater than one."
ϕ(  C::JoeCopula,          t) = 1-(1-exp(-t))^(1/C.θ)
ϕ⁻¹(C::JoeCopula,          t) = -log(1-(1-t)^C.θ)
τ(C::JoeCopula) = 1 - 4sum(1/(k*(2+k*C.θ)*(C.θ*(k-1)+2)) for k in 1:1000) # 446 in R copula. 
radial_dist(C::JoeCopula) = Sibuya(1/C.θ)


