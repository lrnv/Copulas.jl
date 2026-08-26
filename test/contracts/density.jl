function test_density_contract(C, ctx, kind)
    kind === :continuous || return
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    @test p >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    @test loglikelihood(C, ctx.U) isa Real
end
