# Cross-operation family equivalences whose oracle is an alternative representation.

@testset "specialized FGM paths agree with the generic polynomial oracle" begin
    θ = 0.4
    generic = PolynomialOracleCopula(θ)
    specialized = FGMCopula{2}(θ)
    u = [0.37, 0.68]

    generic_integrated_cdf =
        invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, generic, u)
    @test cdf(specialized, u) ≈ generic_integrated_cdf atol=2e-5
    @test pdf(specialized, u) ≈ pdf(generic, u)
    @test Copulas.measure(specialized, [0.15, 0.25], [0.55, 0.65]) ≈
          Copulas.measure(generic, [0.15, 0.25], [0.55, 0.65])

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2])
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6

    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
    @test inverse_rosenblatt(specialized, rosenblatt(specialized, u)) ≈
          inverse_rosenblatt(generic, rosenblatt(generic, u)) atol=2e-6
    @test Copulas.ρ(specialized) ≈ Copulas.ρ(generic) atol=2e-5
    @test Copulas.β(specialized) ≈ Copulas.β(generic)
end

@testset "EV analytic partials agree with the differentiable CDF path" begin
    f(z) = z[1]^2 * z[2]^3 + z[3]
    mixed_point = [0.4, 0.7, 1.1]
    expected12 = 6 * mixed_point[1] * mixed_point[2]^2
    @test Copulas._mixed_partial(f, mixed_point, (1, 2)) ≈ expected12
    @test Copulas._mixed_partial(f, Tuple(mixed_point), [1, 2]) ≈ expected12

    z = [0.31, 0.57, 0.73]
    for C in (LogCopula{3}(2.0), GalambosCopula{3}(0.7))
        analytic = Copulas._partial_cdf(C, (3,), (1, 2),
                                        (z[3],), (z[1], z[2]))
        differentiated = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                b -> cdf(C, [a, b, z[3]]), z[2]), z[1])
        @test analytic ≈ differentiated atol=1e-11 rtol=2e-8
    end

    # Numerical-kernel tails cannot accept dual numbers; their analytic STDF
    # partials must nevertheless power conditioning and Rosenblatt end to end.
    C = tEVCopula{3}(4.0, 0.2)
    D = condition(C, (1, 2), (0.31, 0.58))
    @test 0 < cdf(D, 0.63) < 1
    @test pdf(D, 0.63) > 0
    u = [0.21, 0.53, 0.74]
    @test all(x -> 0 < x < 1, rosenblatt(C, u))
end

@testset "bivariate EV matrix and scalar representations agree" begin
    point = [0.4, 0.7]
    pairs = (
        (HuslerReissCopula{2}([0.0 1.0; 1.0 0.0]),
         HuslerReissCopula{2}(2.0), 4101),
        (tEVCopula{2}(4.0, [1.0 0.3; 0.3 1.0]),
         tEVCopula{2}(4.0, 0.3), 4102),
    )
    for (matrix_model, scalar_model, seed) in pairs
        @test cdf(matrix_model, point) ≈ cdf(scalar_model, point)
        @test pdf(matrix_model, point) ≈ pdf(scalar_model, point)
        for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)
            @test measure(matrix_model) ≈ measure(scalar_model)
        end
        @test rand(Random.Xoshiro(seed), matrix_model, 16) ==
              rand(Random.Xoshiro(seed), scalar_model, 16)
    end
end
