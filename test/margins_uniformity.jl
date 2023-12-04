@testitem "Generic tests on every copulas" begin
    using HypothesisTests, Distributions, Random, WilliamsonTransforms
    using InteractiveUtils
    using ForwardDiff
    using StableRNGs

    cops = (
        IndependentCopula(3),
        AMHCopula(3,0.6),
        AMHCopula(4,-0.3),
        ClaytonCopula(2,-0.7),
        ClaytonCopula(3,-0.1),
        ClaytonCopula(4,7),
        FrankCopula(2,-5),
        FrankCopula(3,12),
        FrankCopula(4,6),
        FrankCopula(4,150),
        JoeCopula(3,7),
        GumbelCopula(4,7),
        GumbelCopula(4,20),
        # GumbelCopula(4,100),
        GumbelBarnettCopula(3,0.7),
        InvGaussianCopula(4,0.05),
        InvGaussianCopula(3,8),
        GaussianCopula([1 0.5; 0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        FGMCopula(2,1),
        MCopula(4),
        WCopula(2),
        ArchimedeanCopula(2,i𝒲(LogNormal(),2)),
        PlackettCopula(2.0),
        EmpiricalCopula(randn(2,100),pseudo_values=false),
        SurvivalCopula(ClaytonCopula(2,-0.7),(1,2)),
        # Others ? Yes probably others too ! 
    )



    #### Try to ensure that every copula in the package is indeed in this list, to remmember contributors to add their model here: 
    function _subtypes(type::Type)
        out = Any[]
        _subtypes!(out, type)
    end
    function _subtypes!(out, type::Type)
        if !isabstracttype(type)
            push!(out, type)
        else
            foreach(T->_subtypes!(out, T), InteractiveUtils.subtypes(type))
        end
        out
    end
    for CT in _subtypes(Copulas.Copula) # Check that every copula type has been used
        @test any(isa(C,CT) for C in cops)
    end
    for TG in _subtypes(Copulas.Generator) # Check that every generator has been used 
        @test any(isa(C.G,TG) for C in cops if typeof(C)<:Copulas.ArchimedeanCopula)
    end


    #### methods to numerically derivate the pdf from the cdf : 
    # Not really efficient as in some cases this return zero while the true pdf is clearly not zero. 
    function _v(u,j,uj)
        return [(i == j ? uj : u[i]) for i in 1:length(u)]
    end
    function _der(j,C,u)
        if j == 1
            return ForwardDiff.derivative(u1 -> cdf(C,_v(u,1,u1)), u[1])
        else
            return ForwardDiff.derivative(uj -> _der(j-1,C,_v(u,j,uj)),u[j])
        end
    end
    function get_numerical_pdf(C,u)
        _der(length(C),C,u)
    end

    # Filter on archimedeans for fitting tests. 
    function is_archimedean_with_agenerator(CT)
        if CT<:ArchimedeanCopula
            GT = Copulas.generatorof(CT)
            if !isnothing(GT)
                if !(GT<:Copulas.ZeroVariateGenerator)
                    if !(GT<:Copulas.WilliamsonGenerator)
                        return true
                    end
                end
            end
        end
        return false
    end
    

    n = 1000
    U = Uniform(0,1)
    for C in cops
        
        d = length(C)
        CT = typeof(C)
        rng = StableRNG(123)
        spl = rand(rng,C,n)

        if !(CT<:TCopula)
        # Check that the cdf has special values at the bounds: 
            @test cdf(C,zeros(d)) == 0
            @test cdf(C,ones(d)) == 1

            # Check that the cdf values are in [0,1]
            @test all(0 .<= cdf(C,spl) .<= 1)
        end
        # Check that samples are in [0,1]:
        @test all(0 <= x <= 1 for x in spl)

        # Check uniformity of each marginal : 
        if !(CT<:EmpiricalCopula) # this one is not a true copula :)
            for i in 1:d
                # On the samples
                @test pvalue(ApproximateOneSampleKSTest(spl[i,:], U),tail=:right) > 0.01 # this is weak but enough to catch mistakes. 

                # On the cdf: 
                u = ones(d)
                for val in [0,1,rand(10)...]
                    u[i] = val
                    if typeof(C)<:TCopula
                        @test_broken cdf(C,u) ≈ val
                    else
                        @test cdf(C,u) ≈ val
                    end
                end
                # extra check for zeros: 
                u = rand(d)
                u[i] = 0
                if typeof(C)<:TCopula
                    @test_broken cdf(C,u) ≈ val
                else
                    @test iszero(cdf(C,u))
                end
            end
        end

        # Conditionally on the applicability of the pdf method... 
        # Finally we do not check pdf, as it is too broken in a lot of cases... 


        # Something should be made to revamp this test 
        # if applicable(pdf,C,spl)

        #     # if archimedean, check also that monotonicity is good: 
        #     if !(CT<:ArchimedeanCopula) || ((Copulas.max_monotony(C.G) > d) && !(typeof(Copulas.williamson_dist(C.G,d))<:WilliamsonTransforms.𝒲₋₁))

        #         # check that pdf values are positives: 
        #         @test all(pdf(C,spl) .>= 0)

        #         # also check that pdf values are indeed derivatives of the cdf values: 
        #         begin 
        #             for _ in 1:10
        #                 u = rand(d)
        #                 @test isapprox(get_numerical_pdf(C,u),pdf(C,u),atol=1e-5)
        #             end
        #         end
        #     end
        # end
        

        # only check archimedeans for tau ∘ tau_inv

        if is_archimedean_with_agenerator(CT)

            if applicable(Copulas.τ,C.G)
                # Check that τ is in [-1,1]:
                tau = Copulas.τ(C)
                @test -1 <= tau <= 1

                # If tau_inv exists, check that it returns the right value here : 
                if applicable(Copulas.τ⁻¹, CT, tau) && is_archimedean_with_agenerator(CT) && applicable(Copulas.τ⁻¹,typeof(C.G),tau)
                    @test Copulas.τ(Copulas.generatorof(CT)(Copulas.τ⁻¹(CT,tau))) ≈ tau
                end
            end

            # Same checks for spearman rho 
            if applicable(Copulas.ρ,C.G)
                # Check that ρ is in [-1,1]:
                rho = Copulas.ρ(C)
                @test -1 <= rho <= 1

                # If tau_inv exists, check that it returns the right value here : 
                if applicable(Copulas.ρ⁻¹, CT, rho) && is_archimedean_with_agenerator(CT) && applicable(Copulas.ρ⁻¹,typeof(C.G),rho)
                    @test Copulas.ρ(Copulas.generatorof(CT)(Copulas.ρ⁻¹(CT,rho))) ≈ rho
                end
            end

            fit(CT,spl)

        end

        # Check that fitting works: 
        # if additional_condition(CT)
            # fit(CT,spl)
        # end
        # @test true

    end 
end