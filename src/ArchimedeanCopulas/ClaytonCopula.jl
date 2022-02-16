struct ClaytonCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
ClaytonCopula(d,θ) = ClaytonCopula{d,typeof(θ)}(θ) # this constructor must be implementeD. 
ϕ(  C::ClaytonCopula,      t) = (1+sign(C.θ)*t)^(-1/C.θ)
ϕ⁻¹(C::ClaytonCopula,      t) = sign(C.θ)*(t^(-C.θ)-1)
radial_dist(C::ClaytonCopula) = Distributions.Gamma(1/C.θ,1)
τ(C::ClaytonCopula) = C.θ/(C.θ+2)
τ⁻¹(::Type{ClaytonCopula},τ) = 2τ/(1-τ)