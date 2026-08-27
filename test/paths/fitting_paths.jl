# Mechanism-path layer: exercises representative Sklar, empirical, covariance,
# optimizer, and model-result fitting routes beyond the universal fit contract.
@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 16)
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau, vcov=false, derived_measures=false)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}

    model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        copula_method=:itau, vcov=false, derived_measures=false)
    @test model.result isa SklarDist
    @test StatsBase.nobs(model) == size(data, 2)

    ecdf_fit = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                   sklar_method=:ecdf, copula_method=:itau, vcov=false,
                   derived_measures=false)
    @test ecdf_fit isa SklarDist
end

@testset "public covariance fitting option" begin
    U = rand(StableRNG(112), ClaytonCopula{2}(1.0), 20)
    model = fit(CopulaModel, ClaytonCopula{2}, U; method=:mle,
                vcov=true, vcov_method=:hessian, derived_measures=false)
    @test StatsBase.vcov(model) isa AbstractMatrix
    @test size(StatsBase.vcov(model)) == (StatsBase.dof(model), StatsBase.dof(model))
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:mle, vcov=true, vcov_method=:invalid, derived_measures=false)
end
