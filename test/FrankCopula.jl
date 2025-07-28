@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,0.5)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,-Inf)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,1.0)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,Inf)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(2,log(rand(M.rng)))) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(3,1.0)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(3,12)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(3,Inf)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,1.0)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,150)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,30)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,37)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,6.)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,6)) end
@testitem "Generic" tags=[:FrankCopula] setup=[M] begin M.check(FrankCopula(4,Inf)) end


@testitem "Frank pdf" begin
    using HCubature
    using Distributions

    for C in [FrankCopula(2, -1.0), FrankCopula(2, 5.0)]
        d = size(C)[1]
        @test hcubature(x -> pdf(C, x), zeros(d), ones(d))[1] ≈ 1.0
    end
end

@testitem "Frank cdf" begin
    using HCubature
    using Distributions

    for C in [FrankCopula(2, -1.0), FrankCopula(2, 5.0)]
        d = size(C)[1]
        u = fill(0.5, d)
        @test hcubature(x -> pdf(C, x), zeros(d), u)[1] ≈ cdf(C, u)
    end
end

@testitem "Frank rand" begin
    using StatsBase

    dimensions = [2, 2, 5]
    parameters = [-1.0, 5.0, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = FrankCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^4)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ atol = 0.1
            end
        end
    end
end

@testitem "Frank derivatives" begin
    using ForwardDiff

    for G in [Copulas.FrankGenerator(5.0), Copulas.FrankGenerator(-1.0)]
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(1), 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈
            Copulas.ϕ⁽ᵏ⁾(G, Val(2), 10.0)

        @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
    end
end

@testitem "Frank Rosenblatt" begin
    using StatsBase

    dimensions = [2, 2, 5]
    parameters = [-1.0, 5.0, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = FrankCopula(d, θ)
        u = rand(C, 1000)
        U = rosenblatt(C, u)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.1
            end
        end
        @test u ≈ inverse_rosenblatt(C, U)
    end
end
