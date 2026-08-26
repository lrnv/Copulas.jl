const PUBLIC_SYMBOLS = (
    :pseudos, :condition, :subsetdims, :rosenblatt, :inverse_rosenblatt, :Nataf,
    :SklarDist, :CopulaModel, :WilliamsonGenerator, :𝒲, :EmpiricalGenerator,
    :DiscreteSpectralTail, :ArchimedeanCopula, :ExtremeValueCopula,
    :LiouvilleCopula, :NestedArchimedeanCopula, :ArchimaxCopula,
    :DiscreteSpectralCopula,
    :AMHCopula, :ClaytonCopula, :FrankCopula, :GumbelCopula,
    :GumbelBarnettCopula, :InvGaussianCopula, :JoeCopula,
    :BB1Copula, :BB2Copula, :BB3Copula, :BB6Copula, :BB7Copula,
    :BB8Copula, :BB9Copula, :BB10Copula,
    :AsymGalambosCopula, :AsymLogCopula, :AsymMixedCopula, :BC2Copula,
    :CuadrasAugeCopula, :EmpiricalEVCopula, :GalambosCopula,
    :HuslerReissCopula, :LogCopula, :MixedCopula, :MOCopula,
    :TawnCopula, :tEVCopula, :BB4Copula, :BB5Copula,
    :GaussianCopula, :TCopula, :BernsteinCopula, :BetaCopula,
    :CheckerboardCopula, :EmpiricalCopula, :FGMCopula,
    :IndependentCopula, :MCopula, :WCopula, :PlackettCopula,
    :RafteryCopula, :SurvivalCopula,
    :Copula, :Distortion, :Generator, :Tail,
    :ϕ, :ϕ⁻¹, :ϕ⁽¹⁾, :ϕ⁻¹⁽¹⁾, :ϕ⁽ᵏ⁾, :ϕ⁽ᵏ⁾⁻¹, :𝒲₋₁, :max_monotony,
    :A, :dA, :d²A, :ℓ, :ellpartial,
    :τ, :ρ, :β, :γ, :ι, :λₗ, :λᵤ, :τ⁻¹, :ρ⁻¹, :β⁻¹, :λᵤ⁻¹,
    :corblomqvist, :corgini, :corentropy, :corlowertail, :coruppertail, :measure,
    :IndependentGenerator, :MGenerator, :WGenerator, :FrailtyGenerator,
    :AMHGenerator, :ClaytonGenerator, :FrankGenerator, :GumbelGenerator,
    :GumbelBarnettGenerator, :InvGaussianGenerator, :JoeGenerator,
    :BB1Generator, :BB2Generator, :BB3Generator, :BB6Generator,
    :BB7Generator, :BB8Generator, :BB9Generator, :BB10Generator,
    :AsymGalambosTail, :AsymLogTail, :AsymMixedTail, :BC2Tail,
    :CuadrasAugeTail, :EmpiricalEVTail, :EmpiricalEVMultivariateTail,
    :GalambosTail, :HuslerReissTail, :LogTail, :MixedTail,
    :MOTail, :TawnTail, :tEVTail,
)

@testset "declared public surface is present" begin
    declared = Set(names(Copulas; all=false, imported=false))
    delete!(declared, :Copulas)
    @test declared == Set(PUBLIC_SYMBOLS)
    for symbol in PUBLIC_SYMBOLS
        @test isdefined(Copulas, symbol)
        @test Base.ispublic(Copulas, symbol)
    end
end
