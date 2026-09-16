@testset "nested fitting reconstruction validates nesting" begin
    template = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(2.0);
        leaves=[1],
        children=[ClaytonCopula{2}(5.0)],
    )
    U = [
        0.20 0.35 0.70 0.55
        0.30 0.65 0.40 0.80
        0.60 0.45 0.75 0.25
    ]

    α = Copulas._nested_unbound(template)
    @test length(α) == 2
    @test isfinite(Copulas._nested_fit_loss(template, α, U))

    # Swapping the two Clayton coordinates makes the parent stronger than the
    # child (θ_parent = 5, θ_child = 2), which violates the certified nesting
    # condition. Objective evaluation must reject that proposal without letting
    # an invalid copula escape, while direct reconstruction must fail normally.
    α_invalid = reverse(α)
    @test isinf(Copulas._nested_fit_loss(template, α_invalid, U))
    @test_throws DomainError Copulas._nested_rebound(template, α_invalid)

    rebuilt = Copulas._nested_rebound(template, α)
    @test rebuilt isa NestedArchimedeanCopula{3}
    @test Copulas._nested_tree_is_certified(rebuilt)
end
