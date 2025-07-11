@testitem "Fix values of bivariate ClaytonCopula: τ, cdf, pdf and contructor" begin
    using Distributions
    using HCubature

    C = ClaytonCopula(2, 2.5)
    @test hcubature(x -> pdf(C, x), zeros(2), ones(2))[1] ≈ 1.0

    # Fix a few cdf and pdf values:
    x = [0:0.25:1;]
    y = x
    cdf1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    cdf2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    pdf2 = [0.0, 2.0, 1.0, 2 / 3, 0.0]
    for i in 1:5
        @test cdf(ClaytonCopula(2, 2), [x[i], y[i]]) ≈ cdf1[i]
        @test cdf(ClaytonCopula(2, -0.5), [x[i], y[i]]) ≈ cdf2[i]
        @test pdf(ClaytonCopula(2, -0.5), [x[i], y[i]]) ≈ pdf2[i]
    end

    # Fix a few tau values:
    @test Copulas.τ(ClaytonCopula(2, -0.5)) == -1 / 3
    @test Copulas.τ(ClaytonCopula(2, 2)) == 0.5
    @test Copulas.τ(ClaytonCopula(2, 10)) == 10 / 12

    # Fix constructor behavior:
    @test isa(ClaytonCopula(2, 0), IndependentCopula)
    @test isa(ClaytonCopula(2, -0.7), ClaytonCopula)
    @test isa(ClaytonCopula(2, -1), WCopula)
    @test isa(ClaytonCopula(2, Inf), MCopula)
end

@testitem "Clayton derivatives" begin
    using ForwardDiff

    for G in [Copulas.ClaytonGenerator(2.0), Copulas.ClaytonGenerator(-0.5)]
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, 1, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈
            Copulas.ϕ⁽ᵏ⁾(G, 2, 10.0)

        @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
    end
end

@testitem "Clayton rand" begin
    using StatsBase

    dimensions = [2, 2, 3, 5]
    parameters = [2.5, -0.5, -0.3, 2.0]

    for (d, θ) in zip(dimensions, parameters)
        C = ClaytonCopula(d, θ)
        u = rand(C, 10^5)
        τ = Copulas.τ(C)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "Clayton Rosenblatt" begin
    using StatsBase

    dimensions = [2, 2, 3, 5]
    parameters = [2.5, -0.5, -0.3, 2.0]

    for (d, θ) in zip(dimensions, parameters)
        C = ClaytonCopula(d, θ)
        u = rand(C, 10^5)
        U = rosenblatt(C, u)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.01
            end
        end
        u = rand(C, 10^4)
        U = rosenblatt(C, u)
        @test u ≈ inverse_rosenblatt(C, U)
    end
end
