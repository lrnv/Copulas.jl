
struct AMHCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
# ϕ(  C::AliMikhailHaq,t) = (1-C.θ)/(exp(t)-C.θ)
# ϕ⁻¹(C::AliMikhailHaq,t) = log((1-C.θ*(1-t))/t)
# # radial_dist(C::AliMikhailHaq) = Distributions.Geometric(1-C.θ)+1

