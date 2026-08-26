function test_conditioning_contract(C, ctx)
    d = length(C)
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
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    @test all(x -> x >= 0, densities)
    @test all(x -> 0 <= x <= 1, rand(StableRNG(73), D, 3))
    q = quantile(D, 0.5)
    @test 0 <= q <= 1
    @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end
