# Operation suite: public contract, generic correctness, specialization
# equivalence, and exhaustive route closure for rectangle probabilities.

function measure_route_key(C)
    Base.@nospecialize C

    return (
        which(
            Copulas.measure,
            Tuple{typeof(C),Vector{Float64},Vector{Float64}},
        ),
        length(C) == 2 ? :bivariate : :multivariate,
    )
end

@testset "measure" begin
    @testset "public contract" begin
        for fixture in COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            d = length(C)
            @testset "$(case.name)" begin
                test_progress("operations", "measure", "contract", case.name)
                @test Copulas.measure(C, zeros(d), ones(d)) ≈ 1 atol=1e-3
                interior = Copulas.measure(C, fill(0.2, d), fill(0.6, d))
                @test 0 <= interior <= 1
            end
        end

        C = ClaytonCopula{2}(1.5)
        @test Copulas.measure(C, [0.7, 0.2], [0.4, 0.8]) == 0
        @test Copulas.measure(C, (0.2, 0.3), (0.7, 0.8)) ≈
              Copulas.measure(C, [0.2, 0.3], [0.7, 0.8])
    end

    @testset "generic correctness" begin
        C = PolynomialOracleCopula{3,Float64}(0.3)
        @test first(measure_route_key(C)) === which(
            Copulas.measure,
            Tuple{Copulas.Copula{3},Vector{Float64},Vector{Float64}},
        )
        lower = [0.12, 0.18, 0.24]
        upper = [0.68, 0.73, 0.81]
        expected = sum(Iterators.product((0:1 for _ in 1:3)...)) do corner
            point = [corner[i] == 1 ? upper[i] : lower[i] for i in 1:3]
            (-1)^(3 - sum(corner)) * _oracle_cdf(C, point)
        end
        @test Copulas.measure(C, lower, upper) ≈ expected atol=2e-8

        split = 0.46
        left_upper = copy(upper)
        left_upper[1] = split
        right_lower = copy(lower)
        right_lower[1] = split
        @test Copulas.measure(C, lower, upper) ≈
              Copulas.measure(C, lower, left_upper) +
              Copulas.measure(C, right_lower, upper) atol=2e-8

        independence = IndependentCopula{3}()
        @test Copulas.measure(independence, lower, upper) ≈ prod(upper - lower)
        @test Copulas.measure(MCopula{2}(), [0.2, 0.2], [0.7, 0.7]) ≈ 0.5
        @test Copulas.measure(WCopula{2}(), [0.2, 0.2], [0.7, 0.7]) ≈ 0.4

        singular = MOCopula{2}(0.2, 0.3, 0.4)
        singular_split = 0.45
        whole = Copulas.measure(singular, [0.1, 0.15], [0.8, 0.75])
        left = Copulas.measure(
            singular, [0.1, 0.15], [singular_split, 0.75])
        right = Copulas.measure(
            singular, [singular_split, 0.15], [0.8, 0.75])
        @test whole ≈ left + right atol=1e-12
    end

    @testset "specialization equivalence and route exhaustiveness" begin
        selected_routes = Set{Any}()
        tested_routes = Set{Any}()
        for fixture in COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            route = measure_route_key(C)
            push!(selected_routes, route)
            route in tested_routes && continue

            d = length(C)
            lower = collect(range(0.13, 0.19; length=d))
            upper = collect(range(0.71, 0.79; length=d))
            expected = 0.0
            for mask in Iterators.product(ntuple(_ -> (false, true), d)...)
                point = [mask[i] ? lower[i] : upper[i] for i in 1:d]
                expected += (-1)^count(identity, mask) * cdf(C, point)
            end
            @testset "$(case.name)" begin
                test_progress("operations", "measure", "route", case.name)
                @test Copulas.measure(C, lower, upper) ≈ expected atol=1e-10
            end
            push!(tested_routes, route)
        end
        @test !isempty(selected_routes)
        @test tested_routes == selected_routes
    end
end
