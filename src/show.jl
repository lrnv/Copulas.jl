function Base.show(io::IO, C::EmpiricalCopula)
    print(io, "EmpiricalCopula{d}$(size(C.u))")
end
function Base.show(io::IO, C::FGMCopula{d, Tθ, Tf}) where {d, Tθ, Tf}
    print(io, "FGMCopula{$d}(θ = $(C.θ))")
end
function Base.show(io::IO, C::SurvivalCopula)
    print(io, "SurvivalCopula($(C.C))")
end
function Base.show(io::IO, C::ArchimedeanCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ExtremeValueCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ArchimaxCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, WilliamsonGenerator{d2, TX}}) where {d, d2, TX}
    print(io, "ArchimedeanCopula($d, i𝒲($(C.G.X), $d2))")
end
function Base.show(io::IO, C::EllipticalCopula)
    print(io, "$(typeof(C))(Σ = $(C.Σ)))")
end
function Base.show(io::IO, G::WilliamsonGenerator{d, TX}) where {d, TX}
    print(io, "i𝒲($(G.X), $(d))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, <:WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}}) where {d, d2}
    print(io, "ArchimedeanCopula($d, EmpiricalGenerator$((d2, length(Distributions.support(C.G.X)))))")
end
function Base.show(io::IO, G::WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}) where {d2}
    print(io, "EmpiricalGenerator$((d2, length(Distributions.support(G.X))))")
end
function Base.show(io::IO, C::SubsetCopula)
    print(io, "SubsetCopula($(C.C), $(C.dims))")
end
function Base.show(io::IO, tail::EmpiricalEVTail)
    print(io, "EmpiricalEVTail(", length(tail.tgrid), " knots)")
end
function Base.show(io::IO, C::ExtremeValueCopula{2, EmpiricalEVTail})
    print(io, "ExtremeValueCopula{2} ⟨", C.tail, "⟩")
end
function Base.show(io::IO, B::BernsteinCopula{d,C}) where {d,C<:Copulas.Copula}
    print(io, "BernsteinCopula{", d, "} ⟨base=", nameof(C), ", m=", B.m, "⟩")
end
function Base.show(io::IO, C::BetaCopula)
    print(io, "BetaCopula{d}$(size(C.ranks))")
end
function Base.show(io::IO, C::CheckerboardCopula{d}) where {d}
    print(io, "CheckerboardCopula{", d, "} ⟨m=", C.m, "⟩")
end