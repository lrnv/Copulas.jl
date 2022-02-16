struct FranckCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
# ϕ(  C::Franck,       t) = -log(1+exp(-t)(exp(-C.θ)-1))/C.θ
# ϕ⁻¹(C::Franck,       t) = -log((exp(-t*C.θ)-1)/(exp(-C.θ)-1))