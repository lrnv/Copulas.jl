@testset "nested template fitting uses dependent Paramorph geometry" begin
    P = Copulas.Paramorph
    C0 = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(2.0);
        children=[ClaytonCopula{2}(5.0)],
    )
    p = P.param_space(C0)
    @test p isa P.DependentProduct
    α0 = P.unconstrain(p, Copulas._nested_parameter_values(C0))
    @test all(isfinite, α0)

    # Arbitrary unconstrained coordinates always reconstruct inside the supported
    # ordering, so no objective-time certificate/Inf barrier is required.
    for α in (zeros(2), [-4.0, -4.0], [3.0, -2.0], [-2.0, 3.0])
        candidate = Copulas._nested_from_coordinates(C0, p, α)
        parent = candidate.G.θ
        child = Copulas._nested_child(only(candidate.children)).G.θ
        @test parent >= 0
        @test child >= parent
    end

    # An invalid template remains constructible but is rejected before fitting.
    bad = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(5.0);
        children=[ClaytonCopula{2}(2.0)],
    )
    U = rand(StableRNG(49_000), C0, 20)
    @test_throws DomainError fit(CopulaModel, bad, U)

    fitted = fit(C0, U)
    @test fitted.G.θ <= Copulas._nested_child(only(fitted.children)).G.θ
end
