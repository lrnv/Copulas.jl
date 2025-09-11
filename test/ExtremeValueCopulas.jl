@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(0.1, [0.2,0.6])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(0.6129496106778634, [0.820474440393214, 0.22304578643880224])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(11.647356700032505, [0.6195348270893413, 0.4197760589260566])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5.0, [0.8, 0.3])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(8.810168494949659, [0.5987759444612732, 0.5391280234619427])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(10+5*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(10+5*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5+4*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5+4*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(rand(M.rng), [rand(M.rng), rand(M.rng)])) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [0.0, 0.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.2, [0.3,0.6])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.5, [0.5, 0.2])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [0.0, 0.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [0.0, 0.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [rand(M.rng), rand(M.rng)])) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.12, 0.13])) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.1, 0.2])) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(1.0, 0.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(0.5, 0.3)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(0.5, 0.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(0.5516353577049822, 0.33689370624999193)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(0.7,0.3)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(1/2,1/2)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :BC2Copula] setup=[M] begin M.check(BC2Copula(rand(M.rng), rand(M.rng))) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.1)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.3437537135972244)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.7103550345192344)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.8)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(1.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(rand(M.rng))) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(0.3)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(120)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(20)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(210)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(4.3)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(80)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(5+5*rand(M.rng))) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(rand(M.rng))) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :GalambosCopula] setup=[M] begin M.check(GalambosCopula(1+4*rand(M.rng))) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :HuslerReissCopula] setup=[M] begin M.check(HuslerReissCopula(0.1)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :HuslerReissCopula] setup=[M] begin M.check(HuslerReissCopula(0.256693308150987)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :HuslerReissCopula] setup=[M] begin M.check(HuslerReissCopula(3.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :HuslerReissCopula] setup=[M] begin M.check(HuslerReissCopula(1.6287031392529938)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :HuslerReissCopula] setup=[M] begin M.check(HuslerReissCopula(5.319851350643586)) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MixedCopula] setup=[M] begin M.check(MixedCopula(0.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MixedCopula] setup=[M] begin M.check(MixedCopula(0.2)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MixedCopula] setup=[M] begin M.check(MixedCopula(0.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MixedCopula] setup=[M] begin M.check(MixedCopula(1.0)) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MOCopula, :OneRing] setup=[M] begin M.check(MOCopula(0.5960710257852946, 0.3313524247810329, 0.09653466861970061)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MOCopula] setup=[M] begin M.check(MOCopula(0.1,0.2,0.3)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MOCopula] setup=[M] begin M.check(MOCopula(0.5, 0.5, 0.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MOCopula] setup=[M] begin M.check(MOCopula(1.0, 1.0, 1.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :MOCopula] setup=[M] begin M.check(MOCopula(rand(M.rng), rand(M.rng), rand(M.rng))) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(10.0, 1.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(3.0, 0.0)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(4.0, 0.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(5.0, -0.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(5.466564460573727, -0.6566645244416698)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(4+6*rand(M.rng), -0.9+1.9*rand(M.rng))) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :tEVCopula] setup=[M] begin M.check(tEVCopula(2.0, 0.5)) end

@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :LogCopula] setup=[M] begin M.check(LogCopula(1+9*rand(M.rng))) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :LogCopula] setup=[M] begin M.check(LogCopula(1.5)) end
@testitem "Generic" tags=[:Generic, :ExtremeValueCopula, :LogCopula] setup=[M] begin M.check(LogCopula(5.5)) end

@testitem "Checking LogCopula == GumbelCopula" tags=[:ExtremeValueCopula, :LogCopula, :ArchimedeanCopula, :GumbelCopula] begin
    # [GenericTests integration]: Probably too specific (equivalence between two constructors/types). Could be a targeted identity test, keep here.
    using InteractiveUtils
    using Copulas, Distributions
    using Random
    using StableRNGs

    rng = StableRNG(1234)
    for θ in [1.0, Inf, 0.5, rand(rng, Uniform(1.0, 10.0))]
        try
            C1 = LogCopula(θ)
            C2 = GumbelCopula(2, θ)
            data = rand(rng, C1, 10)

            for i in 1:10
                u = data[:,i]
                cdf_value_C1 = cdf(C1, u)
                cdf_value_C2 = cdf(C2, u)
                pdf_value_C1 = pdf(C1, u)
                pdf_value_C2 = pdf(C2, u)

                @test isapprox(cdf_value_C1, cdf_value_C2, atol=1e-6) || error("CDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, cdf_value_C1=$cdf_value_C1, cdf_value_C2=$cdf_value_C2")
                @test isapprox(pdf_value_C1, pdf_value_C2, atol=1e-6) || error("PDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, pdf_value_C2=$pdf_value_C1, pdf_value_C2=$pdf_value_C2")
            end
        catch e
            @test e isa ArgumentError
            println("Could not construct LogCopula with θ=$θ: ", e)
        end
    end
end

@testitem "Extreme Galambos density test" tags=[:ExtremeValueCopula, :GalambosCopula] begin
    # [GenericTests integration]: No. This is a trivial smoke test to catch crashes at extreme params; keep as minimal targeted test.
    rand(GalambosCopula(19.7), 1000)
    rand(GalambosCopula(210.0), 1000)
    @test true
end