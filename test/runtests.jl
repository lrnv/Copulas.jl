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

    Copula_Zoo = (
        GumbelCopula(2,1.2),
        ClaytonCopula(4,7.0),
        GaussianCopula([1 0.3; 0.3 1]),
        FrankCopula(5,6.0),
        AMHCopula(3,0.7)
    )

    @testset "testing pdf/cdf values at corners" begin
        for C in Copula_Zoo
            d = length(C)
            @test pdf(C,ones(d)) ≈ 0
            @test pdf(C,zeros(d)) ≈ 0
            @test cdf(C,ones(d)) ≈ 1
            @test cdf(C,zeros(d)) ≈ 0    
        end

        for C in (TCopula(4,[1 0.5; 0.5 1]),)
            d = length(C)
            @test pdf(C,ones(d)) ≈ 0
            @test pdf(C,zeros(d)) ≈ 0
            @test_broken cdf(C,ones(d)) ≈ 1
            @test_broken cdf(C,zeros(d)) ≈ 0    
        end
    end


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
    

    @testset "bare value gaussian model" begin
        # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
        Random.seed!(123)
        C1 = GaussianCopula([1 0.5; 0.5 1]) 
        D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
        @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
    end


    @testset "working measure" begin 
        
        for C in Copula_Zoo
            d = length(C)
            u = zeros(d)
            v = ones(d)

            @test Copulas.measure(C,u,v) >= 0
            
            for i in 1:d
                u[i] = rand()
                v[i] = u[i] + rand()*(1-u[i])
            end
            @test Copulas.measure(C,u,v) >= 0
        end

        for C in (TCopula(4,[1 0.5; 0.5 1]),)
            d = length(C)
            u = zeros(d)
            v = ones(d)

            @test_broken Copulas.measure(C,u,v) >= 0
            
            for i in 1:d
                u[i] = rand()
                v[i] = u[i] + rand()*(1-u[i])
            end
            @test_broken Copulas.measure(C,u,v) >= 0
        end
    end

    # Check that \phi and \phi_inv are indeed inverses of each others. 

    ##########" Tests took from BivairateCopulas.jl


    @testset "Clayton" begin
        
        n = 10^6
        θ = [-0.5, 2, 10]

        @testset "constructor" begin
            @test_broken isa(ClaytonCopula(2,0), Independence)
            @test_throws ClaytonCopula(2,-2)

            @test_logs (:warn, "Clayton returns a W copula for ϑ < -0.5") ClaytonCopula(2,-0.7)
            @test_logs (:warn, "Clayton returns an independence copula for ϑ == 0.0") ClaytonCopula(2,
                0.0
            )
            @test_logs (:warn, "Clayton returns an M copula for ϑ == Inf") ClaytonCopula(2,Inf)
        end

        @testset "generators" begin
            u = [0:0.1:1;]
            for ϑ in θ
                c = ClaytonCopula(2,ϑ)
                @test Copulas.φ⁻¹.(Copulas.φ.(u, c), c) ≈ u
            end
        end

        @testset "τ" begin
            @test τ(ClaytonCopula(2,θ[1])) == -1 / 3
            @test τ(ClaytonCopula(2,θ[2])) == 0.5
            @test τ(ClaytonCopula(2,θ[3])) == 10 / 12
        end

        @testset "sample" begin
            for ϑ in θ
                c = ClaytonCopula(2,ϑ)
                u = rand(c, n)
                @test corkendall(u) ≈ [1.0 τ(c); τ(c) 1.0] atol = 0.01
            end
        end

        # @testset "rosenblatt" begin
        #     for ϑ in θ
        #         c = Clayton(ϑ)
        #         u = rand(c, n)
        #         @test corkendall(rosenblatt(u, c)) ≈ [1.0 0.0; 0.0 1.0] atol = 0.01
        #     end
        # end

        # @testset "inverse_rosenblatt" begin
        #     for ϑ in θ
        #         c = Clayton(ϑ)
        #         u = BivariateCopulas.sample(c, n)

        #         @test inverse_rosenblatt(rosenblatt(u, c), c) ≈ u
        #     end
        # end

        @testset "cdf" begin
            x = [0:0.25:1;]
            y = x

            @test cdf.(ClaytonCopula(2,2), x, y) ≈
                [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
            @test cdf.(ClaytonCopula(2,-0.5), x, y) ≈
                [1.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
        end

        @testset "density" begin
            x = [0:0.25:1;]
            y = x

            @test isnan(pdf(ClaytonCopula(2,2), x[1], y[1]))
            @test pdf.(ClaytonCopula(2,2), x[2:end], y[2:end]) ≈
                [2.2965556205046926, 1.481003649342278, 1.614508582188617, 3.0]

            @test pdf.(ClaytonCopula(2,-0.5), x, y) ≈ [Inf, 2.0, 1.0, 2 / 3, 0.5]
        end
    end

end




