function test_subsetting_contract(C, ctx)
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    point = ctx.u[collect(dims)]
    full_point = ones(d)
    full_point[collect(dims)] = point
    @test cdf(S, point) ≈ cdf(C, full_point)
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (1, 1))
    @test_throws Exception subsetdims(C, (0,))
end
