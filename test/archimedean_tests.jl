
@testitem "Test of τ ∘ τ_inv bijection" begin
    using Random
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

# For each archimedean, we test: 
# - The sampling of a dataset
# - evaluation of pdf and cdf
# - fitting on the dataset. 

@testitem "AMHCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [-1.0,-rand(rng),0.0,rand(rng)]
            C = AMHCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(AMHCopula,data)
        end
    end
    @test true
end
@testitem "ClaytonCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    C = ClaytonCopula(2,-1)
    data = rand(rng,C,100)
    @test all(pdf(C,data) .>= 0)
    @test all(0 .<= cdf(C,data) .<= 1)
    fit(ClaytonCopula,data)
    for d in 2:10
        for θ ∈ [-1/(d-1) * rand(rng),0.0,-log(rand(rng)), Inf]
            C = ClaytonCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(ClaytonCopula,data)
        end
    end
    @test true
end
@testitem "FrankCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    rand(rng,FrankCopula(2,-Inf),10)
    rand(rng,FrankCopula(2,log(rand(rng))),10)

    C = FrankCopula(2,-Inf)
    data = rand(rng,C,100)
    @test all(pdf(C,data) .>= 0)
    @test all(0 .<= cdf(C,data) .<= 1)
    @test_broken fit(FrankCopula,data)

    C = FrankCopula(2,log(rand(rng)))
    data = rand(rng,C,100)
    @test all(pdf(C,data) .>= 0)
    @test all(0 .<= cdf(C,data) .<= 1)
    @test_broken fit(FrankCopula,data)


    for d in 2:10
        for θ ∈ [0.0,rand(rng),1.0,-log(rand(rng)), Inf]
            C = FrankCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            @test_broken fit(FrankCopula,data)
        end
    end
    @test true
end
@testitem "GumbelCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [1.0,1-log(rand(rng)), Inf]
            @show d,θ
            C = GumbelCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(GumbelCopula,data)
        end
    end
    @test true
end
@testitem "JoeCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [1.0,1-log(rand(rng)), Inf]
            C = JoeCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(JoeCopula,data)
        end
    end
    @test true
end

@testitem "GumbelBarnettCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [0.0,rand(rng),1.0]
            C = GumbelBarnettCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            @test_broken fit(GumbelBarnettCopula,data)
        end
    end
    @test true
end

@testitem "InvGaussianCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:10
        for θ ∈ [rand(rng),1.0, -log(rand(rng))]
            @show d,θ
            C = InvGaussianCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            @test_broken fit(InvGaussianCopula,data)
        end
    end
    @test true
end
