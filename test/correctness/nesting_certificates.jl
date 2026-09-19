# Historical filename retained to avoid test-runner churn. There are no
# constructor-level nesting certificates: explicit NAC construction is
# permissive, while template fitting obtains its supported validity geometry
# from Paramorph.

@testset "NAC construction is permissive; fitting geometry is strict" begin
    P = Copulas.Paramorph

    # Same-family ordering can be invalid for the fitting chart without making
    # explicit construction itself an error.
    bad_order = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(5.0);
        children=[ClaytonCopula{2}(2.0)],
    )
    @test bad_order isa NestedArchimedeanCopula
    p = P.param_space(bad_order)
    @test_throws DomainError P.unconstrain(p, Copulas._nested_parameter_values(bad_order))

    # Unsupported fitting geometry is likewise a fit concern, not a constructor
    # concern. The tree remains available for expert/manual use.
    unsupported = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[GumbelCopula{2}(2.0)],
    )
    @test unsupported isa NestedArchimedeanCopula
    @test_throws ArgumentError P.param_space(unsupported)
end
