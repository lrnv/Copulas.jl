@testset "scalar dependence summaries require at least two variables" begin
    U1 = reshape([0.1, 0.3, 0.5, 0.7, 0.9], 1, :)
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ)
        @test_throws DimensionMismatch measure(U1)
    end

    # Entropy is not one of the degenerate concordance normalizations: the
    # data estimator remains mathematically meaningful in one dimension.
    @test Copulas.ι(U1; k=2) isa Real

    U2 = vcat(U1, reverse(U1; dims=2))
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ)
        @test measure(U2) isa Real
    end
end
