function test_distribution_contract(C, ctx)
    d = length(C)
    @test d >= 2
    @test eltype(C) <: Real
    @test params(C) isa NamedTuple
    c = cdf(C, ctx.u)
    @test 0 <= c <= 1
    @test logcdf(C, ctx.u) ≈ log(c)
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test size(ctx.U) == (d, 4)
    @test all(x -> 0 <= x <= 1, ctx.U)
    x = rand(StableRNG(41), C)
    @test length(x) == d
    @test all(y -> 0 <= y <= 1, x)
end
