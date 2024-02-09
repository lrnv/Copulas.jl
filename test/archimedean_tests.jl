
@testitem "Test of τ ∘ τ⁻¹ = Id" begin
    using Random
    using InteractiveUtils
    using StableRNGs
    rng = StableRNG(123)

    inv_works(T,tau) = Copulas.τ(T(2,Copulas.τ⁻¹(T,tau))) ≈ tau
    check_rnd(T,min,max,N) = all(inv_works(T,x) for x in min .+ (max-min) .* rand(rng,N))

    @test check_rnd(ClaytonCopula,       -1,    1,    10)
    @test check_rnd(GumbelCopula,         0,    1,    10)
    @test check_rnd(JoeCopula,            0,    1,    10)
    @test check_rnd(GumbelBarnettCopula, -0.35, 0,    10)
    @test check_rnd(AMHCopula,           -0.18, 0.33, 10)
    @test check_rnd(FrankCopula,         -1,    1,    10)
    @test check_rnd(InvGaussianCopula,    0,    1/2,  10)

    # check we did not forget to add a generator: 
    @test length(InteractiveUtils.subtypes(Copulas.UnivariateGenerator)) == 7
    
end

@testitem "Test of ρ ∘ ρ⁻¹ = Id" begin
    using Random
    using InteractiveUtils
    using StableRNGs
    rng = StableRNG(123)

    inv_works(T,rho) = Copulas.ρ(T(2,Copulas.ρ⁻¹(T,rho))) ≈ rho
    check_rnd(T,min,max,N) = all(inv_works(T,x) for x in min .+ (max-min) .* rand(rng,N))

    # Should be adapted to spearman rho and its inverse when it is possible. 
    @test_broken check_rnd(ClaytonCopula,       -1,    1,    10)
    @test_broken check_rnd(GumbelCopula,         0,    1,    10)
    @test_broken check_rnd(JoeCopula,            0,    1,    10)
    @test_broken check_rnd(GumbelBarnettCopula, -0.35, 0,    10)
    @test_broken check_rnd(AMHCopula,           -0.18, 0.33, 10)
    @test_broken check_rnd(FrankCopula,         -1,    1,    10)
    @test_broken check_rnd(InvGaussianCopula,    0,    1/2,  10)

    # check we did not forget to add a generator: 
    @test length(InteractiveUtils.subtypes(Copulas.UnivariateGenerator)) == 7
end


@testitem "Test of τ against empirical value" begin
    using Random, Distributions
    using StableRNGs
    using StatsBase
    rng = StableRNG(123)

    for C in (
        AMHCopula(3,0.6)          ,
        AMHCopula(4,-0.3)         ,
        ClaytonCopula(2,-0.7)     ,
        ClaytonCopula(3,-0.1)     ,
        ClaytonCopula(4,7)        ,
        FrankCopula(2,-5)         ,
        FrankCopula(3,12)         ,
        FrankCopula(4,6)          ,
        FrankCopula(4,30)         ,
        FrankCopula(4,37)         ,
        FrankCopula(4,150)        ,
        JoeCopula(3,7)            ,
        GumbelCopula(4,7)         ,
        GumbelCopula(4,20)        ,
        GumbelCopula(4,100)       ,
        GumbelBarnettCopula(3,0.7),
        InvGaussianCopula(4,0.05) ,
        InvGaussianCopula(3,8)    ,
    )
        d = length(C)
        K = corkendall(rand(rng,C,1000)')
        avgτ = (sum(K) .- d) / (d^2-d)
        @test avgτ ≈ Copulas.τ(C) atol = 0.03
    end
end

        



@testitem "Fixing simulated values for archimedean copulas" begin
    using Random, Distributions
    using StableRNGs
    rng = StableRNG(123)
    @test rand(rng, AMHCopula(3,0.6)          ) ≈ [0.6100194232653313, 0.2812460189425596, 0.10333020320732342]
    @test rand(rng, AMHCopula(4,-0.3)         ) ≈ [0.6794797200689908, 0.2392452398851621, 0.6025427563179961, 0.2844305227337651]
    @test rand(rng, ClaytonCopula(2,-0.7)     ) ≈ [0.6275520653528824, 0.163616179546588]
    @test rand(rng, ClaytonCopula(3,-0.1)     ) ≈ [0.3881011762518264, 0.5656986324924199, 0.2940237224994986]
    @test rand(rng, ClaytonCopula(4,7)        ) ≈ [0.4939026426425556, 0.5327344049793008, 0.508779929573844, 0.458289642358837]
    @test rand(rng, FrankCopula(2,-5)         ) ≈ [0.7336303867890769, 0.19556711408763564]
    @test rand(rng, FrankCopula(3,12)         ) ≈ [0.705387567921179, 0.8641668940713177, 0.815244058863104]
    @test rand(rng, FrankCopula(4,6)          ) ≈ [0.9003228944757687, 0.8260436103707566, 0.9133247903512886, 0.7994891654043613]
    @test rand(rng, FrankCopula(4,30)         ) ≈ [0.3923476356094088, 0.4382518570496335, 0.39700820876188975, 0.3908555957587312]
    @test rand(rng, FrankCopula(4,37)         ) ≈ [0.7303681473348445, 0.6898271742903759, 0.7264896311455771, 0.7294073988977854]
    @test rand(rng, FrankCopula(4,150)        ) ≈ [0.43642655973527883, 0.42188653827683725, 0.41316130988739547, 0.42762193255859465]
    @test rand(rng, JoeCopula(3,7)            ) ≈ [0.5527859698950577, 0.5839844253671289, 0.6067932545327556]
    @test rand(rng, GumbelCopula(4,7)         ) ≈ [0.781761681793509, 0.7444203095073173, 0.8307664734410207, 0.7495821288234644]
    @test rand(rng, GumbelCopula(4,20)        ) ≈ [0.5482947180691102, 0.5460490383319109, 0.5159827606532166, 0.6002166337095293]
    @test rand(rng, GumbelCopula(4,100)       ) ≈ [0.8199684741926241, 0.81488028576667, 0.8161867411368382, 0.8187239187339411]
    @test rand(rng, GumbelBarnettCopula(3,0.7)) ≈ [0.33125926946816414, 0.31478129636784674, 0.48594013921379653]
    @test rand(rng, InvGaussianCopula(4,0.05) ) ≈ [0.4777472572394352, 0.5714072783562308, 0.8312697205666382, 0.5291602862226205]
    @test rand(rng, InvGaussianCopula(3,8)    ) ≈ [0.5325775326824974, 0.4876131855666078, 0.5207906451760111]
end
  

@testitem "Archimedean - Kendall correlation" begin
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

@testitem "Archimedeans - Spearman correlation" begin

    @test Copulas.ρ(ClaytonCopula(2,3.)) ≈ 0.78645 atol=1.0e-4
    @test Copulas.ρ(ClaytonCopula(2,0.001)) ≈ 0. atol=1.0e-2
    @test Copulas.ρ(GumbelCopula(2,3.)) ≈ 0.8489 atol=1.0e-4

    @test_broken Copulas.ρ⁻¹(ClaytonCopula, 1/3) ≈ 0.58754 atol=1.0e-5
    @test_broken Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0. atol=1.0e-1
    @test_broken Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -.5 atol=1.0e-3
    
    @test_broken Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    @test_broken Copulas.ρ⁻¹(GumbelCopula, 0.0001) == 1.
    
    @test_broken Copulas.ρ⁻¹(FrankCopula, 1/3) ≈ 2.116497 atol=1.0e-5
    @test_broken Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4. atol=1.0e-3

    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.) ≈ 0. atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1.0e-3
end






# For each archimedean, we test: 
# - The sampling of a dataset
# - evaluation of pdf and cdf
# - fitting on the dataset. 

@testitem "AMHCopula - sampling,pdf,cdf,fit" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in 2:5
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
    for d in 2:5
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


    for d in 2:5
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
    for d in 2:5
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
    for d in 2:5
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
    for d in 2:5
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
    for d in 2:5
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
    
    @testset "tail dependencies test" begin
        v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
        v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
        @test tail(v1, v2,  "l", 0.1) ≈ 0.5
        @test tail(v1, v2, "r", 0.1) ≈ 0.5
    end

    @testset "Gumbel" begin
        rng = StableRNG(123)
        x = rand(rng,GumbelCopula(3,2.), 100_000)
        @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
        @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
        @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
        @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.
    end
    @testset "Clayton" begin
        rng = StableRNG(123)
        x = rand(rng,ClaytonCopula(3,1.), 100_000)
        @test_broken tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
        @test_broken tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
        @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    end
    @testset "AMH" begin
        rng = StableRNG(123)
        x = rand(rng,AMHCopula(3,0.8), 100_000)
        @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
        @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    end
    @testset "Frank" begin
        rng = StableRNG(123)
        x = rand(rng,FrankCopula(3,0.8), 100_000)
        @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
        @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    end
end
  
# @testitem "A few tests on bigfloats" begin
#     # using StableRNGs
#     using Random
#     using Distributions
#     using HypothesisTests
#     rng = Random.default_rng() #StableRNG(123) not availiable for bigfloats on old julias.
    
#     for C in (
#         GumbelCopula(3, BigFloat(2.)),
#         ClaytonCopula(2, BigFloat(-.5)),
#         ClaytonCopula(3, BigFloat(2.5)),
#         FrankCopula(2, BigFloat(2.)),
#         AMHCopula(3,BigFloat(.5)),
#         AMHCopula(2,BigFloat(-.3)),
#     )
#         x = rand(rng,C,100)
#         @test_broken typeof(x) == Array{BigFloat,2}
#     end
# end