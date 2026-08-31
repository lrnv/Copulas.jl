# Julia's `export` and `public` declarations are the source of truth for the
# SemVer-governed namespace. Tests derive their cohorts from the module rather
# than maintaining a second list that can drift from the package.
public_symbols() = filter(!=(:Copulas), names(Copulas; all=false, imported=false))

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

@testset "declared public surface is present" begin
    declared = public_symbols()
    @test allunique(declared)
    for symbol in declared
        @test isdefined(Copulas, symbol)
        @test Base.ispublic(Copulas, symbol)
    end
end

@testset verbose=true "every public behaviour is linked to a proof" begin
    @test allunique(getproperty.(PUBLIC_BEHAVIOURS, :name))
    declared_operations = [operation for behaviour in PUBLIC_BEHAVIOURS
                            for operation in behaviour.operations]
    @test allunique(declared_operations)
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
