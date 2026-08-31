# Public contract for copula distribution evaluation. Independent mathematical
# oracles and specialization proofs remain in their historical files until the
# rest of this operation is migrated.

function test_distribution_contract(C, ctx, numerical_atol, margin_atol)
    Base.@nospecialize C ctx
    d = length(C)
    c = cdf(C, ctx.u)
    lower = 0.8 .* ctx.u
    upper = ctx.u .+ 0.2 .* (1 .- ctx.u)

    @test d >= 2
    @test eltype(C) <: Real
    @test params(C) isa NamedTuple
    @test 0 <= c <= 1
    @test max(sum(ctx.u) - d + 1, 0) - 1e-8 <= c <= minimum(ctx.u) + 1e-8
    @test cdf(C, lower) <= c <= cdf(C, upper)
    @test logcdf(C, ctx.u) ≈ log(c) atol=numerical_atol
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test cdf(C, fill(-0.1, d)) == 0
    @test cdf(C, fill(1.1, d)) == 1

    margin = ones(d)
    extended_margin = fill(1.1, d)
    for i in 1:d
        margin .= 1
        extended_margin .= 1.1
        margin[i] = extended_margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=margin_atol
        @test cdf(C, extended_margin) ≈ 0.37 atol=margin_atol
    end

    matrix_u = reshape(ctx.u, :, 1)
    @test cdf(C, matrix_u) ≈ [c] atol=numerical_atol
    @test logcdf(C, matrix_u) ≈ log.([c]) atol=1e-3
    @test_throws ArgumentError cdf(C, zeros(d + 1))
    @test_throws ArgumentError cdf(C, zeros(d + 1, 1))
end

test_density_contract(C, ctx) =
    test_density_contract(Copulas.copula_measure_style(C), C, ctx)
test_density_contract(::Copulas.NonAbsolutelyContinuousMeasure, C, ctx) = nothing
function test_density_contract(::Copulas.AbsolutelyContinuousMeasure, C, ctx)
    Base.@nospecialize C ctx
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    matrix_pdf = pdf(C, reshape(ctx.u, :, 1))

    @test p >= 0
    @test pdf(C, fill(1e-5, length(C))) >= 0
    @test pdf(C, fill(0.5, length(C))) >= 0
    @test pdf(C, fill(1 - 1e-5, length(C))) >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    @test matrix_pdf == [p]
    @test logpdf(C, reshape(ctx.u, :, 1)) ≈ log.(matrix_pdf)
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
    @test_throws DimensionMismatch logpdf(C, zeros(length(C) + 1))
    @test_throws ArgumentError logpdf(C, zeros(length(C) + 1, 1))
end

@testset verbose=true "public distribution-evaluation contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        ctx = copula_contract_context(C, 10_000 + seed)
        test_progress("operations", "distribution", case.name, "cdf")
        test_distribution_contract(C, ctx, case.numerical_atol, case.margin_atol)
        test_progress("operations", "distribution", case.name, "density")
        test_density_contract(C, ctx)
    end
end
