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

    # Construction is permissive about parent-child nesting theory, but every
    # node must still use a generator that is valid at that node's local arity.
    @test_throws DomainError NestedArchimedeanCopula(
        Copulas.WGenerator();
        leaves=[1],
        children=[ClaytonCopula{2}(2.0), ClaytonCopula{2}(3.0)],
    )
    invalid_child = Copulas.NestedArchimedeanCopula{3,typeof(Copulas.WGenerator())}(
        Copulas.WGenerator(), [1, 2, 3], Any[], [1, 2, 3])
    @test_throws DomainError NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[invalid_child],
    )

    # Arbitrary unconstrained coordinates always reconstruct inside the supported
    # ordering, so no objective-time certificate/Inf barrier is required.
    for α in (zeros(2), [-4.0, -4.0], [3.0, -2.0], [-2.0, 3.0])
        candidate = Copulas._nested_from_coordinates(C0, p, α)
        parent = candidate.G.θ
        child = Copulas._nested_child(only(candidate.children)).G.θ
        @test parent >= 0
        @test child >= parent
    end

    # Construction is intentionally permissive. The same-family ordering below
    # is outside the fitting chart, but constructing the explicit tree is valid
    # API and only the Paramorph inverse rejects it.
    bad = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(5.0);
        children=[ClaytonCopula{2}(2.0)],
    )
    @test bad isa NestedArchimedeanCopula
    @test_throws DomainError P.unconstrain(
        P.param_space(bad), Copulas._nested_parameter_values(bad))

    U = rand(StableRNG(49_000), C0, 20)
    @test_throws DomainError fit(CopulaModel, bad, U)

    # Missing fitting geometry is likewise a fit concern, not a constructor
    # concern. Explicit expert/manual trees remain constructible.
    unsupported = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[GumbelCopula{2}(2.0)],
    )
    @test unsupported isa NestedArchimedeanCopula
    @test_throws ArgumentError P.param_space(unsupported)
    @test_throws ArgumentError fit(CopulaModel, unsupported, U)

    fitted = fit(C0, U)
    @test fitted.G.θ <= Copulas._nested_child(only(fitted.children)).G.θ
end
