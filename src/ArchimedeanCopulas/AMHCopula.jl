
struct AMHCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
AMHCopula(d,θ) = (0 <= θ < 1) ? AMHCopula{d,typeof(θ)}(θ) : @error "Theta must be between 0 and 1"
ϕ(  C::AMHCopula,t) = (1-C.θ)/(exp(t)-C.θ)
ϕ⁻¹(  C::AMHCopula,t) = log(C.θ + (1-C.θ)/t)
τ(C::AMHCopula) = 1 - 2(C.θ+(1-C.θ)^2*log(1-C.θ))/(3C.θ^2) # no closed form inverse...
radial_dist(C::AMHCopula) = 1 + Distributions.Geometric(1-C.θ)


