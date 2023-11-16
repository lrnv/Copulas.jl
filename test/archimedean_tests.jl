
@testitem "Test of τ ∘ τ_inv bijection" begin
    using Random
    using StableRNGs
    rng = StableRNG(123)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    for T in (
        # AMHCopula,
        ClaytonCopula,
        # FrankCopula,
        GumbelCopula,
        # IndependentCopula,
        # JoeCopula,
        # GumbelBarnettCopula,
        # InvGaussianCopula
    )
        for τ in taus
            @test Copulas.τ(T(2,Copulas.τ⁻¹(T,τ))) ≈ τ
        end
    end
end


@testitem "AMHCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [-1.0,-rand(rng),0.0,rand(rng)]
            rand(rng,AMHCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "ClaytonCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    rand(rng,ClaytonCopula(2,-1),10)
    for d in 2:10
        for θ ∈ [-1/(d-1) * rand(rng),0.0,-log(rand(rng)), Inf]
            rand(rng,ClaytonCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "FrankCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    rand(rng,FrankCopula(2,-Inf),10)
    rand(rng,FrankCopula(2,log(rand(rng))),10)
    for d in 2:10
        for θ ∈ [0.0,rand(rng),1.0,-log(rand(rng)), Inf]
            rand(rng,FrankCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "GumbelCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [1.0,1-log(rand(rng)), Inf]
            rand(rng,GumbelCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "JoeCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [1.0,1-log(rand(rng)), Inf]
            rand(rng,JoeCopula(d,θ),10)
        end
    end
    @test true
end

@testitem "GumbelBarnettCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [0.0,rand(rng),1.0]
            rand(rng,GumbelBarnettCopula(d,θ),10)
        end
    end
    @test true
end

@testitem "InvGaussianCopula - Test sampling all cases" begin
    using StableRNGs
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [rand(rng),1.0, Inf]
            rand(rng,InvGaussianCopula(d,θ),10)
        end
    end
    @test true
end