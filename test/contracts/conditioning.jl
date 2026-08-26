function test_conditioning_contract(C, ctx)
    D = condition(C, 1, ctx.u[1])
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    q = quantile(D, 0.5)
    @test 0 <= q <= 1
    @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end
