
@testitem "Test of τ ∘ τ_inv bijection" begin
    using Random
    using StableRNGs
    rng = StableRNG(123)

    inv_works(T,tau) = Copulas.τ(T(2,Copulas.τ⁻¹(T,tau))) ≈ tau
    check_rnd(T,min,max,N) = all(inv_works(T,x) for x in min .+ (max-min) .* rand(rng,N))

    @test check_rnd(ClaytonCopula,       -1,    1,   10)
    @test check_rnd(GumbelCopula,         0,    1,   10)
    @test check_rnd(JoeCopula,            0,    1,   10)
    @test check_rnd(GumbelBarnettCopula, -0.35, 0,   10)
    @test check_rnd(AMHCopula,           -0.18, 1/3, 10)
    @test check_rnd(FrankCopula,         -1,    1,   10)
    @test check_rnd(InvGaussianCopula,0,1/2,10)
    
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
    C0 = ClaytonCopula(2,-1)
    data0 = rand(rng,C0,100)
    @test all(pdf(C0,data0) .>= 0)
    @test all(0 .<= cdf(C0,data0) .<= 1)
    fit(ClaytonCopula,data0)
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

    C0 = FrankCopula(2,-Inf)
    data0 = rand(rng,C0,100)
    @test all(pdf(C0,data0) .>= 0)
    @test all(0 .<= cdf(C0,data0) .<= 1)
    fit(FrankCopula,data0)

    C1 = FrankCopula(2,log(rand(rng)))
    data1 = rand(rng,C1,100)
    @test all(pdf(C1,data1) .>= 0)
    @test all(0 .<= cdf(C1,data1) .<= 1)
    fit(FrankCopula,data1)


    for d in 2:10
        for θ ∈ [1.0,1-log(rand(rng)), Inf]
            C = FrankCopula(d,θ)
            data = rand(rng,C,10000)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(FrankCopula,data)
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
            fit(GumbelBarnettCopula,data)
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
            C = InvGaussianCopula(d,θ)
            data = rand(rng,C,100)
            @test all(pdf(C,data) .>= 0)
            @test all(0 .<= cdf(C,data) .<= 1)
            fit(InvGaussianCopula,data)
        end
    end
    @test true
end
