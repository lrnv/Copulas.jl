@testitem "Generic tests on every copulas" begin
    using HypothesisTests, Distributions, Random
    using InteractiveUtils
    using ForwardDiff
    using StableRNGs
    rng = StableRNG(123)
    cops = (
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
        GumbelCopula(4,100),
        GumbelBarnettCopula(3,0.7),
        InvGaussianCopula(4,0.05),
        InvGaussianCopula(3,8),
        GaussianCopula([1 0.5; 0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        FGMCopula(2,1),
        MCopula(4),
        PlackettCopula(2.0),
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
    for CT in _subtypes(Copulas.Copula)
        # CT is now a concrete subtype of Copula. 
        # This will check that 
        @test any(isa(C,CT) for C in cops)
    end

    #### methods to numerically derivate the pdf from the cdf : 
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

    

    n = 1000
    U = Uniform(0,1)
    for C in cops

        d = length(C)
        CT = Float64
        for CC in _subtypes(Copulas.Copula)
            if isa(C,CC)
                CT = CC
                break
            end
        end
        @test CT<:Copulas.Copula # otherwise there is an issue.

        @show C, CT
        
        nfail = 0
        d = length(C)
        spl = rand(rng,C,n)


        # Check that the cdf has special values at the bounds: 
        @test cdf(C,zeros(d)) == 0
        @test cdf(C,ones(d)) == 1

        # Check that the cdf values are in [0,1]
        @test all(0 .<= cdf(C,spl) .<= 1)

        # Check that samples are in [0,1]:
        @test all(0 <= x <= 1 for x in spl)

        # Check uniformity of each marginal : 
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
            @test iszero(cdf(C,u))
        end


        # Conditionally on the applicability of the pdf method... 
        if applicable(pdf,C,spl)
            # check that pdf values are positives: 
            @test all(pdf(C,spl) .>= 0)
            
            # also check that pdf values are indeed derivatives of the cdf values: 
            @test_broken begin 
                for _ in 1:10
                    u = rand(d)
                    get_numerical_pdf(C,u) ≈ pdf(C,u)
                end
            end

        end
        

        additional_condition(CT) = CT<:ArchimedeanCopula ? !isnothing(Copulas.generatorof(CT)) : true

        # Few checks for kendall taux.
        if applicable(Copulas.τ,C)
            # Check that τ is in [-1,1]:
            tau = Copulas.τ(C)
            @test -1 <= tau <= 1

            # If tau_inv exists, check that it returns the right value here : 
            if applicable(Copulas.τ⁻¹, CT, tau) && additional_condition(CT)
                @test Copulas.τ(T(2,Copulas.τ⁻¹(CT,tau))) ≈ tau
            end
        end

        # Same checks for spearman rho 
        if applicable(Copulas.ρ,C)
            # Check that ρ is in [-1,1]:
            rho = Copulas.ρ(C)
            @test -1 <= rho <= 1

            # If tau_inv exists, check that it returns the right value here : 
            if applicable(Copulas.ρ⁻¹, CT, rho) && additional_condition(CT)
                @test Copulas.ρ(T(2,Copulas.ρ⁻¹(CT,rho))) ≈ rho
            end
        end

        # Check that fitting works: 
        if additional_condition(CT)
            fit(CT,spl)
        end
        @test true

    end 
end