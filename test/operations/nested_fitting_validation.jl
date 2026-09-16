@testset "nested template fitting enforces nesting certificates" begin
    C0 = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(2.0);
        children=[ClaytonCopula{2}(5.0)],
    )
    recon = Base.Fix1(Copulas._nested_rebound, C0)
    α0 = Copulas._nested_unbound(C0)

    @test Copulas._nested_fit_candidate(recon, α0) !== nothing

    # For a two-leaf Clayton block, α = log(θ + 1). Move the root above the
    # child so θ_parent > θ_child, which is a certified invalid nesting edge.
    invalid = copy(α0)
    invalid[1] = log(7.0) # θ_parent = 6
    invalid[2] = log(3.0) # θ_child  = 2

    raw = Copulas._nested_rebound(C0, invalid)
    @test !Copulas._nested_tree_certified(raw)
    @test Copulas._nested_fit_candidate(recon, invalid) === nothing
    @test_throws DomainError Copulas._validate_nested_tree(raw)

    # A completed fit must also satisfy the certificates after the minimizer is
    # reconstructed through the ordinary template map.
    U = rand(StableRNG(49_000), C0, 30)
    fitted = fit(C0, U)
    @test Copulas._nested_tree_certified(fitted)
    @test Copulas._validate_nested_tree(fitted) === fitted
end
