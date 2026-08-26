@testset "public nonparametric fitting paths" begin
    data = [0.1 0.3 0.6 0.8 0.2 0.5 0.7 0.9;
            0.8 0.2 0.5 0.7 0.4 0.9 0.1 0.6]
    cases = (
        (EmpiricalCopula, :deheuvels),
        (BetaCopula, :beta),
        (CheckerboardCopula, :exact),
        (BernsteinCopula, :bernstein),
    )
    for (family, method) in cases
        fitted = fit(family, data; method, vcov=false, derived_measures=false)
        @test fitted isa Copulas.Copula{2}
    end

    ev = fit(EmpiricalEVCopula, data; method=:cfg, vcov=false,
             derived_measures=false)
    @test ev isa ExtremeValueCopula{2}
end

@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 16)
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau, vcov=false, derived_measures=false)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}
end
