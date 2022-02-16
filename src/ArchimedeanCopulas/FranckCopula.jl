struct FranckCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
FranckCopula(d,θ) = θ >= 0 ? FranckCopula{d,typeof(θ)}(θ) : @error "Theta must be positive"
ϕ(  C::FranckCopula,       t) = -log(1+exp(-t)*(exp(-C.θ)-1))/C.θ
ϕ⁻¹(C::FranckCopula,       t) = -log((exp(-t*C.θ)-1)/(exp(-C.θ)-1))

D₁ = GSL.sf_debye_1 # sadly, this is C code... corresponds to x -> x^{-1} * \int_0^x (t/(e^t-1)) dt

τ(C::FranckCopula) = 1+4(D₁(C.θ)-1)/C.θ

radial_dist(C::FranckCopula) = Logarithmic(1-exp(-C.θ))


