# Operation suite: public contract, parent-margin correctness, specialized
# representations, value branches, and exhaustive route closure for subsetdims.

function subsetting_route_key(C)
    Base.@nospecialize C

    return (
        which(
            Copulas.subsetdims,
            Tuple{typeof(C),Tuple{Int,Int}},
        ),
        length(C) == 2 ? :bivariate : :multivariate,
    )
end

@testset "subsetdims" begin
    @testset "public contract" begin
        for fixture in COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            d = length(C)
            dims = d == 2 ? (2, 1) : (1, d)
            S = subsetdims(C, dims)
            point = collect(range(0.37, 0.68; length=2))
            parent_point = ones(d)
            parent_point[collect(dims)] = point
            @testset "$(case.name)" begin
                @test length(S) == 2
                @test cdf(S, point) ≈ cdf(C, parent_point) atol=max(1e-5, case.numerical_atol)
                @test length(subsetdims(S, (1,))) == 1
                @test_throws Exception subsetdims(C, (1, 1))
                @test_throws Exception subsetdims(C, (0,))
            end
        end

        C = ClaytonCopula{3}(1.5)
        @test subsetdims(C, [3, 1]) == subsetdims(C, (3, 1))
        first_subset = Copulas.SubsetCopula(C, (3, 1, 2))
        second_subset = Copulas.SubsetCopula(first_subset, (2, 3))
        @test second_subset == subsetdims(C, (1, 2))
    end

    @testset "generic correctness" begin
        C = PolynomialOracleCopula{3,Float64}(0.3)
        dims = (1, 3)
        S = subsetdims(C, dims)
        u = [0.37, 0.68]
        parent_point = [u[1], 1.0, u[2]]
        @test first(subsetting_route_key(C)) === which(
            Copulas.subsetdims,
            Tuple{Copulas.Copula{3},Tuple{Int,Int}},
        )
        @test cdf(S, u) ≈ cdf(C, parent_point)
    end

    @testset "specialization equivalence and route exhaustiveness" begin
        selected_routes = Set{Any}()
        tested_routes = Set{Any}()
        for fixture in COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            route = subsetting_route_key(C)
            push!(selected_routes, route)
            route in tested_routes && continue

            d = length(C)
            dims = d == 2 ? (2, 1) : (1, d)
            S = subsetdims(C, dims)
            u = [0.37, 0.68]
            parent_point = ones(d)
            parent_point[collect(dims)] .= u
            @testset "$(case.name)" begin
                @test cdf(S, u) ≈ cdf(C, parent_point)
            end
            push!(tested_routes, route)
        end
        @test !isempty(selected_routes)
        @test tested_routes == selected_routes
    end

    @testset "full-coordinate permutations" begin
        function permuted_point(perm, u)
            v = similar(u)
            for (i, j) in enumerate(perm)
                v[j] = u[i]
            end
            return v
        end

        C = ClaytonCopula{3}(2.0)
        perm = (2, 3, 1)
        S = subsetdims(C, perm)
        u = [0.31, 0.57, 0.79]
        @test cdf(S, u) ≈ cdf(C, permuted_point(perm, u)) atol=1e-8
        @test logpdf(S, u) ≈ logpdf(C, permuted_point(perm, u)) atol=1e-8

        Σ = [1.0 0.6 0.2; 0.6 1.0 0.5; 0.2 0.5 1.0]
        G = GaussianCopula{3}(Σ)
        permuted = subsetdims(G, perm)
        @test permuted.Σ ≈ Σ[collect(perm), collect(perm)]
        @test logpdf(permuted, u) ≈
              logpdf(G, permuted_point(perm, u)) atol=1e-8
    end

    @testset "Survival coordinate remapping" begin
        u = [0.25, 0.7]
        C3 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (3,))
        subset = subsetdims(C3, (1, 3))
        reference = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
        @test cdf(subset, u) ≈ cdf(reference, u)
        @test pdf(subset, u) ≈ pdf(reference, u)

        C13 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (1, 3))
        reordered = subsetdims(C13, (3, 1))
        reordered_reference = SurvivalCopula{2}(
            ClaytonCopula{2}(2.0), (1, 2))
        @test cdf(reordered, u) ≈ cdf(reordered_reference, u)
        @test pdf(reordered, u) ≈ pdf(reordered_reference, u)
    end
end
