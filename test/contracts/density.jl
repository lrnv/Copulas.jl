function test_density_contract(C, ctx, kind)
    kind === :continuous || return
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    @test p >= 0
    @test pdf(C, fill(1e-5, length(C))) >= 0
    @test pdf(C, fill(0.5, length(C))) >= 0
    @test pdf(C, fill(1 - 1e-5, length(C))) >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    matrix_pdf = pdf(C, reshape(ctx.u, :, 1))
    @test matrix_pdf == [p]
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
end
