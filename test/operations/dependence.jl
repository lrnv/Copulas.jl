# Operation proof for scalar and pairwise dependence measures. During the
# migration, mathematical oracles still live in correctness/ and equivalence/;
# this file already owns the family-wide contract and one execution per route.

function dependence_operation_route_key(measure, C)
    Base.@nospecialize measure C
    method = which(measure, Tuple{typeof(C)})
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

function test_dependence_contract(C)
    Base.@nospecialize C
    # Expensive generic measures compose primitives proved by other operation
    # suites. Applicability is therefore checked for every family, while each
    # selected numerical implementation is executed only once below.
    for measure in SCALAR_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
    for (measure, _) in PAIRWISE_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
end

function test_scalar_dependence_result(measure, C)
    Base.@nospecialize measure C
    value = measure(C)
    @test value isa Real
    @test !isnan(value)
    if measure !== Copulas.ι
        @test -1 <= value <= 1
    end
end

function test_pairwise_dependence_result(measure, diagonal, C)
    Base.@nospecialize measure diagonal C
    d = length(C)
    matrix = measure(C)
    @test size(matrix) == (d, d)
    @test matrix ≈ transpose(matrix)
    @test diag(matrix) == fill(diagonal, d)
    @test all(x -> x isa Real && !isnan(x), matrix)
end

@testset verbose=true "public dependence-measure contract" begin
    @testset "$(fixture.case.name)" for fixture in COPULA_FIXTURES
        test_progress("operations", "dependence", fixture.case.name, "contract")
        test_dependence_contract(fixture.copula)
    end
end

@testset verbose=true "one execution per dependence-measure dispatch" begin
    # Prefer cheap closed-form representatives when several families select
    # the same route. Applicability remains checked for every family above.
    route_cost(case) = case.family in (BernsteinCopula, FGMCopula) ? 0 :
                       case.family === ClaytonCopula ? 1 : 2
    models = sort(collect(ROUTING_COPULA_FIXTURES); by=x -> route_cost(x.case))

    @testset verbose=true "$(nameof(measure))" for measure in SCALAR_DEPENDENCE_MEASURES
        selected_routes = Set(dependence_operation_route_key(measure, fixture.copula)
                              for fixture in models
                              if _dependence_is_defined(measure, fixture.copula))
        tested_routes = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            key = dependence_operation_route_key(measure, copula)
            key in tested_routes && continue
            test_progress("operations", "dependence", nameof(measure), case.name)
            test_scalar_dependence_result(measure, copula)
            push!(tested_routes, key)
        end
        @test tested_routes == selected_routes
    end

    @testset verbose=true "$(nameof(first(entry)))" for entry in PAIRWISE_DEPENDENCE_MEASURES
        measure, diagonal = entry
        selected_routes = Set(dependence_operation_route_key(measure, fixture.copula)
                              for fixture in models
                              if _dependence_is_defined(measure, fixture.copula))
        tested_routes = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            key = dependence_operation_route_key(measure, copula)
            key in tested_routes && continue
            test_progress("operations", "dependence", nameof(measure), case.name)
            test_pairwise_dependence_result(measure, diagonal, copula)
            push!(tested_routes, key)
        end
        @test tested_routes == selected_routes
    end
end
