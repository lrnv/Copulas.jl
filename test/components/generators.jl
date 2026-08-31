# Component proof: exhaustively covers public generator families and
# verifies their transform, inverse, derivative, and reconstruction identities.

@testset "specialized Gumbel generator agrees with its generic oracle" begin
    θ = 1.5
    generic = PowerExponentialOracleGenerator(θ)
    specialized = Copulas.GumbelGenerator(θ)
    for t in (0.2, 0.7, 1.4)
        p = Copulas.ϕ(generic, t)
        @test Copulas.ϕ(specialized, t) ≈ p
        @test Copulas.ϕ⁻¹(specialized, p) ≈ Copulas.ϕ⁻¹(generic, p)
        @test Copulas.ϕ⁽¹⁾(specialized, t) ≈ Copulas.ϕ⁽¹⁾(generic, t)
        @test Copulas.ϕ⁽ᵏ⁾(specialized, 2, t) ≈ Copulas.ϕ⁽ᵏ⁾(generic, 2, t)
        @test Copulas.ϕ⁻¹⁽¹⁾(specialized, p) ≈ Copulas.ϕ⁻¹⁽¹⁾(generic, p)
    end
end
@testset "public generator registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in public_symbols()
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Generator &&
           getfield(Copulas, symbol) <: Copulas.Generator)
    numerical_families = Set(F for F in public_families
                             if !(F <: Copulas.MarkerGenerator))
    represented = Set(typeof(G) for G in GENERATOR_CASES)
    @test all(F -> any(T -> T <: F, represented), numerical_families)
    @test all(T -> any(F -> T <: F, numerical_families), represented)
end

@testset "public generator primitives" begin
    operations = (
        monotonicity = (Copulas.max_monotony, G -> Tuple{typeof(G)}),
        phi = (Copulas.ϕ, G -> Tuple{typeof(G),Float64}),
        inverse = (Copulas.ϕ⁻¹, G -> Tuple{typeof(G),Float64}),
        first = (Copulas.ϕ⁽¹⁾, G -> Tuple{typeof(G),Float64}),
        derivative = (Copulas.ϕ⁽ᵏ⁾, G -> Tuple{typeof(G),Int,Float64}),
        inverse_first = (Copulas.ϕ⁻¹⁽¹⁾, G -> Tuple{typeof(G),Float64}),
        derivative_inverse =
            (Copulas.ϕ⁽ᵏ⁾⁻¹, G -> Tuple{typeof(G),Int,Float64}),
    )
    selected_routes = Dict(name => Set(which(f, signature(G))
        for G in GENERATOR_CASES) for (name, (f, signature)) in pairs(operations))
    checked_routes = Dict(name => Set{Method}() for name in keys(operations))
    for G in GENERATOR_CASES
        @testset "$(nameof(typeof(G)))" begin
            @test G isa Copulas.Generator
            @test Copulas.max_monotony(G) >= 2
            @test params(G) isa NamedTuple
            rebuilt = typeof(G)(values(params(G))...)
            @test params(rebuilt) == params(G)
            @test Copulas.ϕ(G, 0.0) ≈ 1
            @test 0 <= Copulas.ϕ(G, 0.7) <= 1
            p = Copulas.ϕ(G, 0.7)
            @test Copulas.ϕ⁻¹(G, p) ≈ 0.7 atol=2e-6 rtol=2e-6
            @test Copulas.ϕ⁽¹⁾(G, 0.7) <= 0
            @test Copulas.ϕ⁽ᵏ⁾(G, 0, 0.7) ≈ p
            derivative_rtol = G isa WilliamsonGenerator ? 1e-4 : 2e-7
            @test Copulas.ϕ⁽¹⁾(G, 0.7) ≈
                  ForwardDiff.derivative(t -> Copulas.ϕ(G, t), 0.7) rtol=derivative_rtol
            @test Copulas.ϕ⁽ᵏ⁾(G, 1, 0.7) ≈ Copulas.ϕ⁽¹⁾(G, 0.7)
            second_derivative = Copulas.ϕ⁽ᵏ⁾(G, 2, 0.7)
            @test second_derivative >= -sqrt(eps(Float64))
            @test second_derivative ≈
                  ForwardDiff.derivative(t -> Copulas.ϕ⁽¹⁾(G, t), 0.7) rtol=derivative_rtol
            h = 1e-5
            inverse_derivative = (Copulas.ϕ⁻¹(G, 0.5 + h) -
                                  Copulas.ϕ⁻¹(G, 0.5 - h)) / (2h)
            @test Copulas.ϕ⁻¹⁽¹⁾(G, 0.5) ≈ inverse_derivative rtol=2e-5
            y = Copulas.ϕ⁽ᵏ⁾(G, 1, 0.3)
            derivative_inverse = Copulas.ϕ⁽ᵏ⁾⁻¹(G, 1, y)
            @test Copulas.ϕ⁽ᵏ⁾(G, 1, derivative_inverse) ≈ y
            # A Williamson derivative may be flat between radial atoms, so its
            # generalized inverse need not recover the particular input point.
            G isa WilliamsonGenerator ||
                @test derivative_inverse ≈ 0.3 atol=2e-5 rtol=2e-5
            for (name, (f, signature)) in pairs(operations)
                push!(checked_routes[name], which(f, signature(G)))
            end
        end
    end
    @test checked_routes == selected_routes
end

@testset "Williamson inverse dispatch routes" begin
    # Integer and non-integer orders deliberately select different methods.
    # Exercise every route reachable from the public generator registry while
    # keeping one representative per selected Method.
    checked = Dict{Symbol,Set{Method}}(:integer => Set{Method}(),
                                       :real => Set{Method}())
    selected = Dict(
        :integer => Set(which(Copulas.𝒲₋₁, Tuple{typeof(G),Int})
                        for G in GENERATOR_CASES),
        :real => Set(which(Copulas.𝒲₋₁, Tuple{typeof(G),Float64})
                     for G in GENERATOR_CASES),
    )
    for G in GENERATOR_CASES
        for (kind, order) in ((:integer, 2), (:real, 1.5))
            method = which(Copulas.𝒲₋₁, Tuple{typeof(G),typeof(order)})
            method in checked[kind] && continue
            radial = Copulas.𝒲₋₁(G, order)
            @test radial isa Distributions.UnivariateDistribution
            @test minimum(radial) >= 0
            push!(checked[kind], method)
        end
    end
    @test checked == selected
end
