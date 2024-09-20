@testitem "Generic tests on every copulas" begin
    using HypothesisTests, Distributions, Random, WilliamsonTransforms
    using InteractiveUtils
    using ForwardDiff
    using StatsBase: corkendall
    using StableRNGs
    rng = StableRNG(123)

    bestiary = unique([
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
        EmpiricalCopula(randn(2,2000),pseudo_values=false),
        SurvivalCopula(ClaytonCopula(2,-0.7),(1,2)),
        RafteryCopula(2, 0.2),
        RafteryCopula(3, 0.5),
        AsymGalambosCopula(5.0, [0.8, 0.3]),
        AsymLogCopula(1.5, [0.5, 0.2]),
        AsymMixedCopula([0.1, 0.2]),
        BC2Copula(0.5, 0.3),
        CuadrasAugeCopula(0.8),
        GalambosCopula(4.3),
        HuslerReissCopula(3.5),
        LogCopula(5.5),
        MixedCopula(0.5),
        MOCopula(0.1, 0.2, 0.3),
        tEVCopula(4.0, 0.5),
        Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2,1)),
        AsymGalambosCopula(0.1, [0.2,0.6]),
        AsymLogCopula(1.2, [0.3,0.6]),
        AsymMixedCopula([0.1,0.2]),
        BC2Copula(0.7,0.3),
        CuadrasAugeCopula(0.1),
        GalambosCopula(0.3),
        HuslerReissCopula(0.1),
        LogCopula(1.5),
        MixedCopula(0.2),
        MOCopula(0.1,0.2,0.3),
        tEVCopula(2.0,0.5),
        IndependentCopula(2),
        AsymGalambosCopula(0.6129496106778634, [0.820474440393214, 0.22304578643880224]),
        AsymGalambosCopula(8.810168494949659, [0.5987759444612732, 0.5391280234619427]),
        AsymGalambosCopula(11.647356700032505, [0.6195348270893413, 0.4197760589260566]),
        AsymLogCopula(1.0, [0.8360692316060747, 0.68704221750134]),
        AsymLogCopula(1.0, [0.0, 0.0]),
        AsymLogCopula(1.0, [1.0, 1.0]),
        AsymLogCopula(2.8130363753722403, [0.3539590866764071, 0.15146985093210463]),
        AsymLogCopula(2.8130363753722403, [0.0, 0.0]),
        AsymLogCopula(2.8130363753722403, [1.0, 1.0]),
        AsymLogCopula(12.29006035397328, [0.7036713552821277, 0.7858058549340399]),
        AsymLogCopula(12.29006035397328, [0.0, 0.0]),
        AsymLogCopula(12.29006035397328, [1.0, 1.0]),
        AsymMixedCopula([0.1, 0.2]),
        AsymMixedCopula([0.2, 0.4]),
        GalambosCopula(0.6129496106778634),
        GalambosCopula(8.810168494949659),
        GalambosCopula(11.647356700032505),
        GalambosCopula(20),
        GalambosCopula(60),
        GalambosCopula(70),
        GalambosCopula(80),
        GalambosCopula(120),
        GalambosCopula(210),
        GalambosCopula(0.40543796744015514),
        GalambosCopula(2.675150743283436),
        GalambosCopula(6.730938346629261),
        BC2Copula(0.5516353577049822, 0.33689370624999193),
        BC2Copula(1.0, 0.0),
        BC2Copula(0.5, 0.5),
        CuadrasAugeCopula(0.7103550345192344),
        CuadrasAugeCopula(0.3437537135972244),
        MCopula(2),
        HuslerReissCopula(0.256693308150987),
        HuslerReissCopula(1.6287031392529938),
        HuslerReissCopula(5.319851350643586),
        MixedCopula(1.0),
        MixedCopula(0.2),
        MixedCopula(0.5),
        MOCopula(0.1, 0.2, 0.3),
        MOCopula(1.0, 1.0, 1.0),
        MOCopula(0.5, 0.5, 0.5),
        MOCopula(0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
        tEVCopula(2.0, 0.5),
        tEVCopula(5.0, -0.5),
        tEVCopula(5.466564460573727, -0.6566645244416698),
        LogCopula(4.8313231991648244),
        [
            [AMHCopula(d,θ) for d in 2:4 for θ ∈ [-1.0,-rand(rng),0.0,rand(rng)]]...,
            ClaytonCopula(2,-1),
            [ClaytonCopula(d,θ) for d in 2:4 for θ ∈ [-1/(d-1) * rand(rng),0.0,-log(rand(rng)), Inf]]...,
            FrankCopula(2,-Inf),
            FrankCopula(2,log(rand(rng))),
            [FrankCopula(d,θ) for d in 2:4 for θ ∈ [1.0,1-log(rand(rng)), Inf]]...,
            [GumbelCopula(d,θ) for d in 2:4 for θ ∈ [1.0,1-log(rand(rng)), Inf]]...,
            [JoeCopula(d,θ) for d in 2:4 for θ ∈ [1.0,1-log(rand(rng)), Inf]]...,
            [GumbelBarnettCopula(d,θ) for d in 2:4 for θ ∈ [0.0,rand(rng),1.0]]...,
            [InvGaussianCopula(d,θ) for d in 2:4 for θ ∈ [rand(rng),1.0, -log(rand(rng))]]...,
        ]...
    ])

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

    # Check that every copula type is indeed represented: 
    @test all(any(isa(C,CT) for C in bestiary) for CT in _subtypes(Copulas.Copula))
     # Check that every archimedean generator has been used 
    @test all(any(isa(C.G,TG) for C in bestiary if typeof(C)<:Copulas.ArchimedeanCopula) for TG in _subtypes(Copulas.Generator))
    
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

    n = 10
    for C in bestiary
        d = length(C)
        CT = typeof(C)
        spl = rand(rng,C,n)

        # Test that cdf returns 0 and 1 at zero(d) and one(d), and is between 0 and 1 inside: 
        @test iszero(cdf(C,zeros(d)))
        @test isone(cdf(C,ones(d)))
        @test 0 <= cdf(C,rand(rng,d)) <= 1
        @test all(0 .<= spl .<= 1)

        # Check CDF marginal uniformity: 
        if !(CT<:EmpiricalCopula) # this one is not a true copula :)
            for i in 1:d
                for val in rand(rng,5)
                    u = ones(d)
                    u[i] = val
                    @test cdf(C,u) ≈ val atol=1e-5
                end
                # extra check for zeros: 
                u = rand(rng,d)
                u[i] = 0
                @test iszero(cdf(C,u))
            end
        end

        # Check that pdf is positive: 
        has_pdf(C) = applicable(Distributions._logpdf,C,rand(length(C),3))
        if has_pdf(C)
            @test pdf(C,ones(length(C))/2) >= 0
            @test all(pdf(C, spl) .>= 0)
        end

        # Check that τ matches empirical values: 
        K = corkendall(rand(rng,C,10000)')
        Kth = corkendall(C)
        @test all(-1 .<= Kth .<= 1)
        rrrr = all(isapprox.(Kth,K; atol=0.1))
        if !rrrr
            @show C
            display(K)
            display(Kth)
        end
        @test rrrr

        # Extra checks, only for archimedeans. 
        if is_archimedean_with_agenerator(CT)
            if applicable(Copulas.τ,C.G)
                # Check that τ is in [-1,1]:
                tau = Copulas.τ(C)

                # If tau_inv exists, check that it returns the right value here : 
                if applicable(Copulas.τ⁻¹, CT, tau) && applicable(Copulas.τ⁻¹,typeof(C.G),tau)
                    @test Copulas.τ(Copulas.generatorof(CT)(Copulas.τ⁻¹(CT,tau))) ≈ tau
                end
            end

            # Same checks for spearman rho 
            if applicable(Copulas.ρ,C.G)
                # Check that ρ is in [-1,1]:
                rho = Copulas.ρ(C)
                @test -1 <= rho <= 1

                # If tau_inv exists, check that it returns the right value here : 
                if applicable(Copulas.ρ⁻¹, CT, rho) && applicable(Copulas.ρ⁻¹,typeof(C.G),rho)
                    @test Copulas.ρ(Copulas.generatorof(CT)(Copulas.ρ⁻¹(CT,rho))) ≈ rho
                end
            end
            # Check that the fit procedure does run:
            fit(CT,spl)
        end
    end 
end
