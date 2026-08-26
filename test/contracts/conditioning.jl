function test_conditioning_contract(C, ctx)
    d = length(C)
    if d > 2
        joint = condition(C, 1, ctx.u[1])
        @test length(joint) == d - 1
        @test 0 <= cdf(joint, ctx.u[2:end]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(ctx.u[1:(d - 1)])
    D = condition(C, js, values)
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    q = quantile(D, 0.5)
    @test 0 <= q <= 1
    @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end
