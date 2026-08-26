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
    for i in 1:d
        margin = ones(d)
        margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=1e-6
    end
    @test cdf(C, reshape(ctx.u, :, 1)) == [c]
    @test measure(C, zeros(d), ones(d)) ≈ 1
    @test measure(C, fill(0.2, d), fill(0.6, d)) >= 0
    @test size(ctx.U) == (d, 4)
    @test all(x -> 0 <= x <= 1, ctx.U)
    x = rand(StableRNG(41), C)
    @test length(x) == d
    @test all(y -> 0 <= y <= 1, x)
    @test_throws ArgumentError cdf(C, zeros(d + 1))
    @test_throws ArgumentError cdf(C, zeros(d + 1, 1))
end
