# Public-API contract: mechanically fixes the complete exported and `public`
# namespace, so adding or removing a SemVer-governed symbol requires a test edit.
const PUBLIC_SYMBOLS = (
    :pseudos, :condition, :subsetdims, :rosenblatt, :inverse_rosenblatt, :Nataf,
    :SklarDist, :CopulaModel, :WilliamsonGenerator, :𝒲, :EmpiricalGenerator,
    :DiscreteSpectralTail, :ArchimedeanCopula, :ExtremeValueCopula,
    :LiouvilleCopula, :NestedArchimedeanCopula, :ArchimaxCopula,
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

# Public methods adopted from other packages do not appear in `names(Copulas)`.
# Keep their behavioural contracts explicit and link every behaviour to the
# test layers that establish availability, correctness, and route coverage.
const PUBLIC_BEHAVIOURS = (
    (name=:construction,
     operations=(:constructors, :params, :length, :eltype),
     contracts=("constructors.jl", "public_compositions.jl"),
     proofs=("mathematical.jl",), routes=("constructors.jl",)),
    (name=:distribution,
     operations=(:cdf, :logcdf, :pdf, :logpdf, :loglikelihood, :rand, :rand!),
     contracts=("copulas.jl", "sklar.jl"),
     proofs=("mathematical.jl", "statistical.jl"), routes=("dispatch.jl",)),
    (name=:subsetting,
     operations=(:subsetdims,), contracts=("copulas.jl",),
     proofs=("mathematical.jl",), routes=("dispatch.jl",)),
    (name=:conditioning,
     operations=(:condition, :quantile),
     contracts=("copulas.jl", "distortions.jl", "sklar.jl"),
     proofs=("mathematical.jl",), routes=("dispatch.jl",)),
    (name=:rosenblatt,
     operations=(:rosenblatt, :inverse_rosenblatt), contracts=("copulas.jl",),
     proofs=("mathematical.jl", "statistical.jl"), routes=("dispatch.jl",)),
    (name=:dependence,
     operations=(:τ, :ρ, :β, :γ, :ι, :λₗ, :λᵤ, :corkendall,
                 :corspearman, :corblomqvist, :corgini, :corentropy,
                 :corlowertail, :coruppertail, :measure,
                 :τ⁻¹, :ρ⁻¹, :β⁻¹, :λᵤ⁻¹),
     contracts=("copulas.jl", "utilities.jl"),
     proofs=("mathematical.jl", "measure_inverses.jl"),
     routes=("dispatch.jl", "measure_inverses.jl")),
    (name=:fitting,
     operations=(:fit, :dof, :nobs, :coef, :coefnames,
                 :deviance, :nullloglikelihood, :nulldeviance, :isfitted,
                 :vcov, :stderror, :confint, :aic, :bic, :residuals, :predict),
     contracts=("fitting.jl",), proofs=("measure_inverses.jl",),
     routes=("fitting.jl",)),
    (name=:generators,
     operations=(:ϕ, :ϕ⁻¹, :ϕ⁽¹⁾, :ϕ⁻¹⁽¹⁾, :ϕ⁽ᵏ⁾, :ϕ⁽ᵏ⁾⁻¹,
                 :𝒲₋₁, :max_monotony),
     contracts=("public_compositions.jl", "univariate_distributions.jl"),
     proofs=("generators.jl", "mathematical.jl"), routes=("generators.jl",)),
    (name=:tails,
     operations=(:A, :dA, :d²A, :ℓ, :ellpartial),
     contracts=("public_compositions.jl",),
     proofs=("tails.jl", "mathematical.jl"), routes=("tails.jl",)),
    (name=:nataf,
     operations=(:Nataf,), contracts=("nataf.jl", "utilities.jl"),
     proofs=("specializations.jl",), routes=("specializations.jl",)),
    (name=:utilities,
     operations=(:pseudos,), contracts=("utilities.jl",),
     proofs=("mathematical.jl",), routes=("utilities.jl",)),
    (name=:extensions,
     operations=(:package_extensions,), contracts=("extensions",),
     proofs=("extensions",), routes=("extensions",)),
)

# Executable transcription of the behavioural table in docs/api/public.md.
# The equality below prevents an operation from being added to the declared
# SemVer contract without being assigned all four proof obligations above.
const DOCUMENTED_PUBLIC_OPERATIONS = Set((
    :constructors, :params, :length, :eltype,
    :cdf, :logcdf, :pdf, :logpdf, :loglikelihood, :rand, :rand!,
    :subsetdims, :condition, :quantile, :rosenblatt, :inverse_rosenblatt,
    :τ, :ρ, :β, :γ, :ι, :λₗ, :λᵤ, :corkendall, :corspearman,
    :corblomqvist, :corgini, :corentropy, :corlowertail, :coruppertail,
    :τ⁻¹, :ρ⁻¹, :β⁻¹, :λᵤ⁻¹, :measure,
    :fit, :dof, :nobs, :coef, :coefnames, :deviance,
    :nullloglikelihood, :nulldeviance, :isfitted, :vcov, :stderror,
    :confint, :aic, :bic, :residuals, :predict,
    :ϕ, :ϕ⁻¹, :ϕ⁽¹⁾, :ϕ⁻¹⁽¹⁾, :ϕ⁽ᵏ⁾, :ϕ⁽ᵏ⁾⁻¹, :𝒲₋₁,
    :max_monotony, :A, :dA, :d²A, :ℓ, :ellpartial,
    :Nataf, :pseudos, :package_extensions,
))

@testset "declared public surface is present" begin
    declared = Set(names(Copulas; all=false, imported=false))
    delete!(declared, :Copulas)
    @test declared == Set(PUBLIC_SYMBOLS)
    for symbol in PUBLIC_SYMBOLS
        @test isdefined(Copulas, symbol)
        @test Base.ispublic(Copulas, symbol)
    end
end

@testset verbose=true "every public behaviour is linked to a proof" begin
    @test allunique(getproperty.(PUBLIC_BEHAVIOURS, :name))
    declared_operations = [operation for behaviour in PUBLIC_BEHAVIOURS
                            for operation in behaviour.operations]
    @test allunique(declared_operations)
    @test Set(declared_operations) == DOCUMENTED_PUBLIC_OPERATIONS
    contract_dir = @__DIR__
    correctness_dir = joinpath(dirname(contract_dir), "correctness")
    routing_dir = joinpath(dirname(contract_dir), "routing")
    equivalence_dir = joinpath(dirname(contract_dir), "equivalence")
    for behaviour in PUBLIC_BEHAVIOURS
        @testset "$(behaviour.name)" begin
            @test !isempty(behaviour.operations)
            @test !isempty(behaviour.contracts)
            @test !isempty(behaviour.proofs)
            @test !isempty(behaviour.routes)
            for file in behaviour.contracts
                file == "extensions" || @test isfile(joinpath(contract_dir, file))
            end
            for file in behaviour.proofs
                file == "extensions" && continue
                @test isfile(joinpath(correctness_dir, file)) ||
                      isfile(joinpath(equivalence_dir, file))
            end
            for file in behaviour.routes
                file == "extensions" && continue
                @test isfile(joinpath(routing_dir, file)) ||
                      isfile(joinpath(equivalence_dir, file)) ||
                      isfile(joinpath(correctness_dir, file)) ||
                      isfile(joinpath(contract_dir, file))
            end
        end
    end
end
