@testitem "fitting archimedians" begin
    
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123)
    MyD = SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta()))
    u = rand(rng,MyD,10000)
    rand!(MyD,u)
    fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
    fit(SklarDist{GaussianCopula,Tuple{LogNormal,Pareto,Beta}},u)
    @test 1==1
    # loglikelyhood(MyD,u)
end

# We could test loklikelyhood for every copula on a standard uniform sample. 
# We should also test the fit function on several sklar models. 
# and teszt the loglikelyhood of the SlakrDist. 

# We should also test other htings ? Dunno what yet. 
# We could also test the behavior of Turing models, so that what Herb did will not fade away with releases; 


# @testitem "GaussianCopula" begin
#     C = GaussianCopula([1 -0.1; -0.1 1])
#     M1 = Beta(2,3)
#     M2 = LogNormal(2,3)
#     D = SklarDist(C,(M1,M2))
#     X = rand(rng,D,1000)
#     loglikelihood(D,X)
#     fit(SklarDist{TCopula,Tuple{Beta,LogNormal}},X) # should give a very high \nu for the student copula. 
# end

# Same thing with other models ? 



@testitem "testing cdf values at corners" begin
    using Distributions
    using Random

    Copula_Zoo = (
        GumbelCopula(2,1.2),
        ClaytonCopula(4,7.0),
        GaussianCopula([1 0.3; 0.3 1]),
        FrankCopula(5,6.0),
        AMHCopula(3,0.7)
    )


    for C in Copula_Zoo
        d = length(C)
       
        @test cdf(C,ones(d)) ≈ 1
        @test cdf(C,zeros(d)) ≈ 0    
    end

    for C in (TCopula(4,[1 0.5; 0.5 1]),)
        d = length(C)
        @test_broken cdf(C,ones(d)) ≈ 1
        @test_broken cdf(C,zeros(d)) ≈ 0    
    end
end

@testitem "pdf/cdf archimedean" begin
    using Distributions
    using Random

    x = Normal(0,1); y = Normal(0,2);
    C = GumbelCopula(2, 1.2)  # a type of Archimedean copula
    D = SklarDist(C, (x,y))
    
    pdf(D, ([1.0, 1.0]))
    cdf(D, ([1.0, 1.0]))
    @test 1==1
end

@testitem "pdf/cdf gaussian" begin
    using Distributions
    using Random

    x = Normal(0, 1)
    y = Normal(0, 2)
    C = GaussianCopula([1 0.5; 0.5 1])
    D = SklarDist(C, (x,y))
    
    pdf(D, ([1.0, 1.0])) # this is fine
    cdf(D, ([1.0, 1.0])) # now passes.
    @test 1==1
end

@testitem "pdf/cdf student" begin
    using Distributions
    using Random

    x = Normal(0, 1)
    y = Normal(0, 2)
    C = TCopula(4,[1 0.5; 0.5 1])
    D = SklarDist(C, (x,y))

    pdf(D, ([1.0, 1.0])) # this is fine
    cdf(D, ([1.0, 1.0])) # this produces error due to non-existance of cdf of multivariate student in Distributions.jl
    @test 1==1
end


@testitem "bare value gaussian model" begin
    using Distributions
    using Random

    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula([1 0.5; 0.5 1]) 
    D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
end


@testitem "working measure" begin 
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123)
    
    for C in (ClaytonCopula(4,7.0),GumbelCopula(2, 1.2))
        d = length(C)
        u = zeros(d)
        v = ones(d)

        @test Copulas.measure(C,u,v) >= 0
        
        for i in 1:d
            u[i] = rand(rng)
            v[i] = u[i] + rand(rng)*(1-u[i])
        end
        @test Copulas.measure(C,u,v) >= 0
    end

    for C in (TCopula(4,[1 0.5; 0.5 1]),)
        d = length(C)
        u = zeros(d)
        v = ones(d)

        @test_broken Copulas.measure(C,u,v) >= 0
        
        for i in 1:d
            u[i] = rand(rng)
            v[i] = u[i] + rand(rng)*(1-u[i])
        end
        @test_broken Copulas.measure(C,u,v) >= 0
    end
end
    



