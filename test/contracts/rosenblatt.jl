function test_rosenblatt_contract(C, ctx, invertible)
    R = rosenblatt(C, ctx.U)
    @test size(R) == size(ctx.U)
    @test all(x -> 0 <= x <= 1, R)
    invertible || return
    @test inverse_rosenblatt(C, R) ≈ ctx.U atol=2e-5 rtol=2e-5
    @test rosenblatt(C, ctx.u) ≈ vec(rosenblatt(C, reshape(ctx.u, :, 1)))
    @test inverse_rosenblatt(C, rosenblatt(C, ctx.u)) ≈ ctx.u atol=2e-5 rtol=2e-5
end
