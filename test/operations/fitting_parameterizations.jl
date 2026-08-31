# Fitting-operation proof for parameterizations. Public
# route availability and result interfaces are covered by contracts/routing;
# this file checks that unconstrained coordinates map bijectively to the
# intended constrained parameter space.
@testset "asymmetric Mixed feasible fitting parameterization" begin
    for (i, z) in pairs(([-3.0, 3.0], [3.0, -3.0], [0.0, 0.5]))
        p = Copulas._rebound_params(Copulas.AsymMixedTail, 2, z)
        i == 1 && @test p.θ₂ > 0
        i == 2 && @test p.θ₂ < 0
        @test p.θ₁ >= 0
        @test p.θ₁ + p.θ₂ <= 1
        @test p.θ₁ + 2p.θ₂ <= 1
        @test p.θ₁ + 3p.θ₂ >= 0
        @test Copulas._unbound_params(Copulas.AsymMixedTail, 2, p) ≈ z
        @test Copulas.AsymMixedTail(p.θ₁, p.θ₂) isa Copulas.AsymMixedTail
    end

    # The reverse direction starts from independently chosen feasible model
    # parameters, so this is not merely a circular composition of one map.
    for p in ((; θ₁=0.25, θ₂=0.10), (; θ₁=1.20, θ₂=-0.30))
        restored = Copulas._rebound_params(Copulas.AsymMixedTail, 2,
            Copulas._unbound_params(Copulas.AsymMixedTail, 2, p))
        @test restored.θ₁ ≈ p.θ₁ atol=3e-11 rtol=3e-11
        @test restored.θ₂ ≈ p.θ₂ atol=3e-11 rtol=3e-11
    end

    example = Copulas._example(Copulas.AsymMixedCopula, 2)
    p = params(example)
    @test example isa Copulas.AsymMixedCopula
    @test keys(p) == (:θ₁, :θ₂)
    @test !iszero(p.θ₂)
end
