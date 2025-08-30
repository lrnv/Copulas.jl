@testitem "Generic" tags=[:ArchimedeanCopula] setup=[M] begin using Distributions; M.check(ArchimedeanCopula(2,i𝒲(LogNormal(),2))) end

@testitem "Archimedean - Fix Kendall correlation" begin
    using Random
    using StableRNGs
    rng = StableRNG(123)

    @test Copulas.Debye(0.5,1) ≈ 0.8819271567906056
    @test Copulas.τ⁻¹(FrankCopula, 0.6) ≈ 7.929642284264058
    @test Copulas.τ⁻¹(GumbelCopula, 0.5) ≈ 2.
    @test Copulas.τ⁻¹(ClaytonCopula, 1/3) ≈ 1.
    @test Copulas.τ⁻¹(AMHCopula, 1/4) ≈ 0.8384520912688538
    @test Copulas.τ⁻¹(AMHCopula, 0.) ≈ 0.
    @test Copulas.τ⁻¹(AMHCopula, 1/3+0.0001) ≈ 1.
    @test Copulas.τ⁻¹(AMHCopula, -2/11) ≈ -1.
    @test Copulas.τ⁻¹(AMHCopula, -0.1505) ≈ -0.8 atol=1.0e-3
    @test Copulas.τ⁻¹(FrankCopula, -0.3881) ≈ -4. atol=1.0e-3
    @test Copulas.τ⁻¹(ClaytonCopula, -1/3) ≈ -.5 atol=1.0e-5
end

@testitem "Archimedeans - Fix Spearman correlation" begin

    @test Copulas.ρ(ClaytonCopula(2,3.)) ≈ 0.78645 atol=1.0e-4
    @test Copulas.ρ(ClaytonCopula(2,0.001)) ≈ 0. atol=1.0e-2
    @test Copulas.ρ(GumbelCopula(2,3.)) ≈ 0.8489 atol=1.0e-4

    @test Copulas.ρ⁻¹(ClaytonCopula, 1/3) ≈ 0.58754 atol=1.0e-5
    @test Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0. atol=1.0e-1
    @test Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -.5 atol=1.0e-3

    @test Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    @test Copulas.ρ⁻¹(GumbelCopula, 0.0001) ≈ 1. atol=1.0e-3

    @test Copulas.ρ⁻¹(FrankCopula, 1/3) ≈ 2.116497 atol=1.0e-5
    @test Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4. atol=1.0e-3

    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test Copulas.ρ⁻¹(AMHCopula, 0.) ≈ 0. atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1.0e-3
end

@testitem "Testing empirical tail values of certain copula samples" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)

    function tail(v1::Vector{T}, v2::Vector{T}, tail::String, α::T = 0.002) where T <: Real
        if tail == "l"
            return sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)
        elseif tail == "r"
            return sum((v1 .> (1-α)) .* (v2 .> (1-α)))./(length(v1)*α)
        end
        0.
    end

    # tail dependencies test
    v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
    v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
    @test tail(v1, v2,  "l", 0.1) ≈ 0.5
    @test tail(v1, v2, "r", 0.1) ≈ 0.5

    # Gumbel
    rng = StableRNG(123)
    x = rand(rng,GumbelCopula(3,2.), 100_000)
    @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.

    # Clayton
    rng = StableRNG(123)
    x = rand(rng,ClaytonCopula(3,1.), 100_000)
    @test_broken tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test_broken tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

    # AMH
    rng = StableRNG(123)
    x = rand(rng,AMHCopula(3,0.8), 100_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    
    # Frank
    rng = StableRNG(123)
    x = rand(rng,FrankCopula(3,0.8), 100_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end