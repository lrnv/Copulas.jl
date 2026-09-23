@testset "nested template fitting uses dependent Paramorph geometry" begin
    P = Copulas.Paramorph
    C0 = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(2.0);
        children=[ClaytonCopula{2}(5.0)],
    )
    α0 = Copulas._nested_unbound(C0)
    @test all(isfinite, α0)

    # An explicit independence parent carries no scalar parameter of its own.
    # Its supported :free edge must therefore delegate directly to the child's
    # local geometry rather than attempting to read a parent parameter.
    I0 = NestedArchimedeanCopula(
        Copulas.IndependentGenerator();
        leaves=[1],
        children=[ClaytonCopula{2}(2.0)],
    )
    αI = Copulas._nested_unbound(I0)
    @test length(αI) == 1
    @test all(isfinite, αI)
    Iroundtrip = Copulas._nested_rebound(I0, αI)
    @test Iroundtrip.G isa Copulas.IndependentGenerator
    @test Copulas._nested_child(only(Iroundtrip.children)).G.θ ≈ 2.0

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
        candidate = Copulas._nested_rebound(C0, α)
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
    @test_throws DomainError Copulas._nested_unbound(bad)

    U = rand(StableRNG(49_000), C0, 20)
    @test_throws DomainError fit(CopulaModel, bad, U)

    # Missing fitting geometry is likewise a fit concern, not a constructor
    # concern. Explicit expert/manual trees remain constructible.
    unsupported = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[GumbelCopula{2}(2.0)],
    )
    @test unsupported isa NestedArchimedeanCopula
    @test_throws ArgumentError Copulas._nested_unbound(unsupported)
    @test_throws ArgumentError fit(CopulaModel, unsupported, U)

    fitted = fit(C0, U)
    @test fitted.G.θ <= Copulas._nested_child(only(fitted.children)).G.θ
end
