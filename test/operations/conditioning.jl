# Public contract for marginal distortions and conditional copulas. Independent
# identities and specialization proofs remain in the historical conditioning
# files until the complete operation is consolidated here.

function test_conditioning_contract(C, ctx)
    Base.@nospecialize C ctx
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
        js = Tuple(1:(d - 2))
        joint = condition(C, js, Tuple(ctx.u[1:(d - 2)]))
        @test length(joint) == 2
        @test 0 <= cdf(joint, ctx.u[(d - 1):d]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(ctx.u[1:(d - 1)])
    D = condition(C, js, values)
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    q = quantile(D, 0.5)

    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
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
    @test 0 <= q <= 1
    is_absolutely_continuous(C) &&
        @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end

@testset verbose=true "public conditioning contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        test_progress("operations", "conditioning", fixture.case.name, "contract")
        test_conditioning_contract(
            fixture.copula,
            copula_contract_context(fixture.copula, 10_000 + seed),
        )
    end
end
