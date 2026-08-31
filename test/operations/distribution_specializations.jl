# Distribution-operation equivalence proofs for optimized CDF and density routes.

function _unique_distribution_routes(operation, predicate)
    Base.@nospecialize operation predicate
    seen = Set{Method}()
    routes = NamedTuple[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        predicate(case, C) || continue
        method = operation(case, C)
        method in seen && continue
        push!(seen, method)
        push!(routes, (; case, C, method))
    end
    return routes
end

@testset verbose=true "specialized continuous CDFs agree with density integration" begin
    routes = _unique_distribution_routes(
        (_, C) -> which(Copulas._cdf, Tuple{typeof(C),Vector{Float64}}),
        (case, C) -> is_absolutely_continuous(C) &&
            !(C isa Union{CheckerboardCopula,LiouvilleCopula}),
    )
    generic_method = which(Copulas._cdf,
        Tuple{Copulas.Copula,Vector{Float64}})
    compared = 0
    u = [0.53, 0.67]
    for (; case, C, method) in routes
        if method === generic_method
            # The generic density integral is independently validated by the
            # polynomial oracle in correctness/mathematical.jl.
            prove_dispatch_route!(:cdf, C, :generic_density_integral)
            continue
        end
        @testset "$(case.name)" begin
            test_progress("equivalence", "cdf", case.name)
            expected = if C isa ArchimedeanCopula
                Copulas.ϕ(C.G, sum(Copulas.ϕ⁻¹(C.G, x) for x in u))
            else
                invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
            end
            @test isapprox(cdf(C, u), expected;
                           atol=max(3e-5, case.numerical_atol), rtol=3e-5)
        end
        prove_dispatch_route!(:cdf, C,
                              C isa ArchimedeanCopula ?
                              :generator_composition : :density_integration)
        compared += 1
    end
    @test compared > 0
end

@testset "checkerboard CDF equals exact box overlap" begin
    fixture = only(filter(x -> x.copula isa CheckerboardCopula,
                          ROUTING_COPULA_FIXTURES))
    case, C = fixture.case, fixture.copula
    u = [0.53, 0.67]
    expected = zero(eltype(values(C.boxes)))
    for (box, weight) in C.boxes
        overlap = one(expected)
        for i in eachindex(u)
            overlap *= clamp(C.m[i] * u[i] - box[i], 0, 1)
        end
        expected += weight * overlap
    end
    @test cdf(C, u) ≈ expected
    prove_dispatch_route!(:cdf, C, :exact_box_overlap)
end

@testset verbose=true "specialized bivariate log-densities agree with CDF derivatives" begin
    routes = _unique_distribution_routes(
        (_, C) -> which(Distributions._logpdf,
                        Tuple{typeof(C),Vector{Float64}}),
        (case, C) -> is_absolutely_continuous(C) && !(C isa LiouvilleCopula),
    )
    u = [0.53, 0.67]
    h = 2e-5
    for (; case, C, method) in routes
        @testset "$(case.name)" begin
            test_progress("equivalence", "logpdf", case.name)
            expected = (
                cdf(C, u .+ (h, h)) - cdf(C, u .+ (h, -h)) -
                cdf(C, u .+ (-h, h)) + cdf(C, u .- (h, h))
            ) / (4h^2)
            @test isapprox(pdf(C, u), expected; atol=8e-4, rtol=8e-4)
            @test logpdf(C, u) ≈ log(pdf(C, u))
        end
        prove_dispatch_route!(:logpdf, C, :cdf_mixed_derivative)
    end
    @test !isempty(routes)
end

@testset "singular and mixed CDF routes satisfy mass identities" begin
    seen = Set{Any}()
    split = 0.46
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        is_absolutely_continuous(C) && continue
        key = dispatch_route_key(:cdf, C)
        key in seen && continue
        push!(seen, key)
        d = length(C)
        for i in 1:d
            margin_point = ones(d)
            margin_point[i] = 0.37
            @test cdf(C, margin_point) ≈ 0.37 atol=case.margin_atol
        end
        lower = collect(range(0.12, 0.18; length=d))
        upper = collect(range(0.78, 0.84; length=d))
        whole = Copulas.measure(C, lower, upper)
        left_upper = copy(upper)
        left_upper[1] = split
        right_lower = copy(lower)
        right_lower[1] = split
        @test whole ≈
              Copulas.measure(C, lower, left_upper) +
              Copulas.measure(C, right_lower, upper)
        prove_dispatch_route!(:cdf, C, :singular_mass_identity)
    end
    @test !isempty(seen)
end
