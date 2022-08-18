using Copulas
using Test
using Distributions
using Random

@testset "Copulas.jl" begin

    @testset "fitting archimedians" begin
        MyD = SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta()))
        u = rand(MyD,10000)
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


    # @testset "GaussianCopula" begin
    #     C = GaussianCopula([1 -0.1; -0.1 1])
    #     M1 = Beta(2,3)
    #     M2 = LogNormal(2,3)
    #     D = SklarDist(C,(M1,M2))
    #     X = rand(D,1000)
    #     loglikelihood(D,X)
    #     fit(SklarDist{TCopula,Tuple{Beta,LogNormal}},X) # should give a very high \nu for the student copula. 
    # end

    # Same thing with other models ? 

    @testset "pdf/cdf archimedean" begin
        x = Normal(0,1); y = Normal(0,2);
        C = GumbelCopula(2, 1.2)  # a type of Archimedean copula
        D = SklarDist(C, (x,y))
        
        pdf(D, ([1.0, 1.0]))
        cdf(D, ([1.0, 1.0]))
        @test 1==1
    end
    
    @testset "pdf/cdf gaussian" begin
        x = Normal(0, 1)
        y = Normal(0, 2)
        C = GaussianCopula([1 0.5; 0.5 1])
        D = SklarDist(C, (x,y))
        
        pdf(D, ([1.0, 1.0])) # this is fine
        cdf(D, ([1.0, 1.0])) # now passes.
        @test 1==1
    end
    
    @testset "pdf/cdf student" begin
        x = Normal(0, 1)
        y = Normal(0, 2)
        C = TCopula(4,[1 0.5; 0.5 1])
        D = SklarDist(C, (x,y))
    
        pdf(D, ([1.0, 1.0])) # this is fine
        cdf(D, ([1.0, 1.0])) # this produces error due to non-existance of cdf of multivariate student in Distributions.jl
        @test 1==1
    end
    
end




