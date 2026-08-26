function test_subsetting_contract(C, ctx)
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (0,))
end
