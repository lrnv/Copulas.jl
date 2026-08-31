# Public-API contract: applies the universal distribution, sampling, subsetting,
# conditioning, Rosenblatt, and dependence-measure behavior to every family.
struct CopulaContractContext{TU,TM}
    u::TU
    U::TM
end

# Deliberately omits the matrix sampler required from concrete copulas. It
# exercises the generic developer-facing diagnostic without duplicating a
# public operation implementation.
struct MissingSamplerContractCopula <: Copulas.Copula{2} end

@testset "copula measure-style trait" begin
    discrete_radial = WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 3)
    @test Copulas.copula_measure_style(ArchimedeanCopula{3}(discrete_radial)) isa
          Copulas.NonAbsolutelyContinuousMeasure
    # Marginalization multiplies the preserved radial by a continuous beta
    # variable, so a positive discrete source becomes absolutely continuous.
    @test Copulas.copula_measure_style(ArchimedeanCopula{2}(discrete_radial)) isa
          Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(
        ArchimedeanCopula{2}(WilliamsonGenerator(Uniform(1.0, 2.0), 2)),
    ) isa Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(ClaytonCopula{3}(-0.5)) isa
          Copulas.NonAbsolutelyContinuousMeasure

    discrete_liouville = LiouvilleCopula{2}(
        WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 2), (0.8, 1.2),
    )
    @test Copulas.copula_measure_style(discrete_liouville) isa
          Copulas.NonAbsolutelyContinuousMeasure
    discrete_archimax = ArchimaxCopula{2}(
        WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 2),
        Copulas.GalambosTail(1.0),
    )
    @test Copulas.copula_measure_style(discrete_archimax) isa
          Copulas.NonAbsolutelyContinuousMeasure

    singular = RafteryCopula{3}(0.5)
    @test Copulas.copula_measure_style(
        Copulas.SubsetCopula(singular, (1, 2)),
    ) isa Copulas.NonAbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(
        SurvivalCopula{3}(singular, (1,)),
    ) isa Copulas.NonAbsolutelyContinuousMeasure
end

function copula_contract_context(C, seed)
    Base.@nospecialize C
    d = length(C)
    u = collect(range(0.31, 0.69; length=d))
    U = rand(StableRNG(seed), C, 4)
    return CopulaContractContext{typeof(u),typeof(U)}(u, U)
end

function test_distribution_contract(C, ctx, numerical_atol, margin_atol)
    Base.@nospecialize C
    Base.@nospecialize ctx
    d = length(C)
    @test d >= 2
    @test eltype(C) <: Real
    @test params(C) isa NamedTuple
    c = cdf(C, ctx.u)
    @test 0 <= c <= 1
    @test max(sum(ctx.u) - d + 1, 0) - 1e-8 <= c <= minimum(ctx.u) + 1e-8
    lower = 0.8 .* ctx.u
    upper = ctx.u .+ 0.2 .* (1 .- ctx.u)
    @test cdf(C, lower) <= c <= cdf(C, upper)
    @test logcdf(C, ctx.u) ≈ log(c) atol=numerical_atol
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test cdf(C, fill(-0.1, d)) == 0
    @test cdf(C, fill(1.1, d)) == 1
    for i in 1:d
        margin = ones(d)
        margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=margin_atol
        extended_margin = fill(1.1, d)
        extended_margin[i] = 0.37
        @test cdf(C, extended_margin) ≈ 0.37 atol=margin_atol
    end
    matrix_u = reshape(ctx.u, :, 1)
    @test cdf(C, matrix_u) ≈ [c] atol=numerical_atol
    @test logcdf(C, matrix_u) ≈ log.([c]) atol=1e-3
    @test Copulas.measure(C, zeros(d), ones(d)) ≈ 1 atol=1e-3
    @test Copulas.measure(C, fill(0.2, d), fill(0.6, d)) >= 0
    @test size(ctx.U) == (d, 4)
    @test eltype(ctx.U) == eltype(C)
    @test all(x -> 0 <= x <= 1, ctx.U)
    buffer = zeros(eltype(C), d, 2)
    @test rand!(StableRNG(40), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    x = rand(StableRNG(41), C)
    @test length(x) == d
    @test eltype(x) == eltype(C)
    @test all(y -> 0 <= y <= 1, x)
    @test_throws ArgumentError cdf(C, zeros(d + 1))
    @test_throws ArgumentError cdf(C, zeros(d + 1, 1))
end

test_density_contract(C, ctx) =
    test_density_contract(Copulas.copula_measure_style(C), C, ctx)
test_density_contract(::Copulas.NonAbsolutelyContinuousMeasure, C, ctx) = nothing
function test_density_contract(::Copulas.AbsolutelyContinuousMeasure, C, ctx)
    Base.@nospecialize C
    Base.@nospecialize ctx
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    @test p >= 0
    @test pdf(C, fill(1e-5, length(C))) >= 0
    @test pdf(C, fill(0.5, length(C))) >= 0
    @test pdf(C, fill(1 - 1e-5, length(C))) >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    matrix_pdf = pdf(C, reshape(ctx.u, :, 1))
    @test matrix_pdf == [p]
    @test logpdf(C, reshape(ctx.u, :, 1)) ≈ log.(matrix_pdf)
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
    @test_throws DimensionMismatch logpdf(C, zeros(length(C) + 1))
    @test_throws ArgumentError logpdf(C, zeros(length(C) + 1, 1))
end

function test_subsetting_contract(C, ctx, numerical_atol)
    Base.@nospecialize C
    Base.@nospecialize ctx
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    point = ctx.u[collect(dims)]
    full_point = ones(d)
    full_point[collect(dims)] = point
    @test cdf(S, point) ≈ cdf(C, full_point) atol=max(1e-5, numerical_atol)
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (1, 1))
    @test_throws Exception subsetdims(C, (0,))
end

function test_conditioning_contract(C, ctx)
    Base.@nospecialize C
    Base.@nospecialize ctx
    d = length(C)
    if d == 2
        scalar = condition(C, 1, ctx.u[1])
        tupled = condition(C, (1,), (ctx.u[1],))
        @test scalar isa Distributions.UnivariateDistribution
        @test cdf(scalar, ctx.u[2]) ≈ cdf(tupled, ctx.u[2])
    end
    if d > 2
        joint = condition(C, 1, ctx.u[1])
        @test length(joint) == d - 1
        @test 0 <= cdf(joint, ctx.u[2:end]) <= 1
    end
    if d > 3
        js2 = Tuple(1:(d - 2))
        joint2 = condition(C, js2, Tuple(ctx.u[1:(d - 2)]))
        @test length(joint2) == 2
        @test 0 <= cdf(joint2, ctx.u[(d - 1):d]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(ctx.u[1:(d - 1)])
    D = condition(C, js, values)
    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    @test logcdf(D, 0.5) ≈ log(cdf(D, 0.5))
    if is_absolutely_continuous(C)
        densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
        @test all(x -> x >= 0, densities)
        density = pdf(D, 0.5)
        @test iszero(density) ? logpdf(D, 0.5) == -Inf :
              logpdf(D, 0.5) ≈ log(density)
    end
    @test all(x -> 0 <= x <= 1, rand(StableRNG(73), D, 3))
    q = quantile(D, 0.5)
    @test 0 <= q <= 1
    # Continuous conditionals invert their CDF. For mixed/singular models the
    # public quantile convention is only required to return a valid support
    # point; atom semantics are checked in `correctness/mathematical.jl`.
    is_absolutely_continuous(C) &&
        @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end

function test_rosenblatt_contract(C, ctx)
    Base.@nospecialize C
    Base.@nospecialize ctx
    R = rosenblatt(C, ctx.U)
    @test size(R) == size(ctx.U)
    @test all(x -> 0 <= x <= 1, R)
    @test rosenblatt(C, ctx.u) ≈ vec(rosenblatt(C, reshape(ctx.u, :, 1)))
    test_rosenblatt_inverse_contract(Copulas.copula_measure_style(C), C, ctx, R)
end

test_rosenblatt_inverse_contract(
    ::Copulas.NonAbsolutelyContinuousMeasure, C, ctx, R,
) = nothing
function test_rosenblatt_inverse_contract(
    ::Copulas.AbsolutelyContinuousMeasure, C, ctx, R,
)
    @test inverse_rosenblatt(C, R) ≈ ctx.U atol=2e-5 rtol=2e-5
    @test inverse_rosenblatt(C, rosenblatt(C, ctx.u)) ≈ ctx.u atol=2e-5 rtol=2e-5
end

_dependence_is_defined(measure, C::Copulas.Copula) =
    _dependence_is_defined(measure, Copulas.copula_measure_style(C))
_dependence_is_defined(
    ::Union{typeof(Copulas.ι),typeof(Copulas.corentropy)},
    ::Copulas.NonAbsolutelyContinuousMeasure,
) = false
_dependence_is_defined(::Any, ::Copulas.CopulaMeasureStyle) = true
function _dependence_dispatch_key(measure, C)
    Base.@nospecialize measure C
    return (which(measure, Tuple{typeof(C)}),
            length(C) == 2 ? :bivariate : :multivariate)
end

function test_dependence_contract(C)
    Base.@nospecialize C
    # Distribution, density, sampling and subsetting primitives are exercised
    # above for every family.  The expensive generic measures only compose
    # those primitives, so the per-family API contract needs to guarantee that
    # dispatch exists; each distinct implementation is executed once below.
    for measure in SCALAR_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
    for (measure, _) in PAIRWISE_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
end

function test_scalar_dependence_result(measure, C)
    Base.@nospecialize measure C
    value = measure(C)
    @test value isa Real
    @test !isnan(value)
    if measure !== Copulas.ι
        @test -1 <= value <= 1
    end
end

function test_pairwise_dependence_result(measure, diagonal, C)
    Base.@nospecialize measure diagonal C
    d = length(C)
    matrix = measure(C)
    @test size(matrix) == (d, d)
    @test matrix ≈ transpose(matrix)
    @test diag(matrix) == fill(diagonal, d)
    @test all(x -> x isa Real && !isnan(x), matrix)
end

function test_copula_contract(case, C, seed)
    Base.@nospecialize case
    Base.@nospecialize C
    ctx = copula_contract_context(C, seed)
    @testset "distribution" begin
        test_progress("contracts", "copulas", case.name, "distribution")
        test_distribution_contract(C, ctx, case.numerical_atol, case.margin_atol)
    end
    @testset "density" begin
        test_progress("contracts", "copulas", case.name, "density")
        test_density_contract(C, ctx)
    end
    @testset "subsetting" begin
        test_progress("contracts", "copulas", case.name, "subsetting")
        test_subsetting_contract(C, ctx, case.numerical_atol)
    end
    @testset "conditioning" begin
        test_progress("contracts", "copulas", case.name, "conditioning")
        test_conditioning_contract(C, ctx)
    end
    @testset "Rosenblatt" begin
        test_progress("contracts", "copulas", case.name, "Rosenblatt")
        test_rosenblatt_contract(C, ctx)
    end
    @testset "dependence" begin
        test_progress("contracts", "copulas", case.name, "dependence")
        test_dependence_contract(C)
    end
end

@testset "public copula registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in PUBLIC_SYMBOLS
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Copula &&
           getfield(Copulas, symbol) <: Copulas.Copula)
    represented = Set(typeof(fixture.copula) for fixture in COPULA_FIXTURES)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset verbose=true "public copula contract" begin
    @testset verbose=true "$(COPULA_CASES[i].name)" for i in eachindex(COPULA_CASES)
        (; case, copula) = COPULA_FIXTURES[i]
        test_copula_contract(case, copula, 10_000 + i)
    end
end

@testset "collection adapters preserve the public semantics" begin
    C = ClaytonCopula{3}(1.5)
    u = [0.3, 0.5, 0.7]
    @test Base.broadcastable(C)[] === C
    @test subsetdims(C, [3, 1]) == subsetdims(C, (3, 1))
    @test cdf(condition(C, [1], [u[1]]), u[2:3]) ≈
          cdf(condition(C, (1,), (u[1],)), u[2:3])

    @test_throws ArgumentError rand!(
        StableRNG(42), MissingSamplerContractCopula(), zeros(2, 1))

    # Repeated subsetting composes coordinate maps relative to the original.
    first_subset = Copulas.SubsetCopula(C, (3, 1, 2))
    second_subset = Copulas.SubsetCopula(first_subset, (2, 3))
    @test second_subset == subsetdims(C, (1, 2))
end

@testset verbose=true "one execution per dependence-measure dispatch" begin
    # Several families can select the exact same adapter.  Prefer cheap,
    # closed-form representatives for that one execution; applicability is
    # still checked independently for every public family above.
    # Bernstein selects the unbranched generic `Copula` measures while its
    # polynomial CDF is much cheaper to integrate than Liouville's numerical
    # CDF. Liouville's family-specific radial identities remain independently
    # proved in the correctness layer.
    route_cost(case) = case.name == "Bernstein" ? 0 :
                       case.name == "FGM" ? 0 :
                       case.name == "Clayton" ? 1 : 2
    models = sort(collect(ROUTING_COPULA_FIXTURES); by=x -> route_cost(x.case))

    @testset verbose=true "$(nameof(measure))" for measure in SCALAR_DEPENDENCE_MEASURES
        seen = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            method, dimension_path = _dependence_dispatch_key(measure, copula)
            (method, dimension_path) in seen && continue
            push!(seen, (method, dimension_path))
            @testset "$(case.name)" begin
                test_progress("contracts", "dependence", nameof(measure), case.name)
                test_scalar_dependence_result(measure, copula)
            end
        end
    end

    @testset verbose=true "$(nameof(first(entry)))" for entry in PAIRWISE_DEPENDENCE_MEASURES
        measure, diagonal = entry
        seen = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            method, dimension_path = _dependence_dispatch_key(measure, copula)
            (method, dimension_path) in seen && continue
            push!(seen, (method, dimension_path))
            @testset "$(case.name)" begin
                test_progress("contracts", "dependence", nameof(measure), case.name)
                test_pairwise_dependence_result(measure, diagonal, copula)
            end
        end
    end
end
