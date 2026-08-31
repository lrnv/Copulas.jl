# Complete operation proof for copula distribution evaluation: public CDF and
# density contracts, independent identities, specialization equivalence, and
# exhaustive dispatch-route registration.

function test_distribution_contract(C, ctx, numerical_atol, margin_atol)
    Base.@nospecialize C ctx
    d = length(C)
    c = cdf(C, ctx.u)
    lower = 0.8 .* ctx.u
    upper = ctx.u .+ 0.2 .* (1 .- ctx.u)

    @test d >= 2
    @test eltype(C) <: Real
    @test params(C) isa NamedTuple
    @test 0 <= c <= 1
    @test max(sum(ctx.u) - d + 1, 0) - 1e-8 <= c <= minimum(ctx.u) + 1e-8
    @test cdf(C, lower) <= c <= cdf(C, upper)
    @test logcdf(C, ctx.u) ≈ log(c) atol=numerical_atol
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test cdf(C, fill(-0.1, d)) == 0
    @test cdf(C, fill(1.1, d)) == 1

    margin = ones(d)
    extended_margin = fill(1.1, d)
    for i in 1:d
        margin .= 1
        extended_margin .= 1.1
        margin[i] = extended_margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=margin_atol
        @test cdf(C, extended_margin) ≈ 0.37 atol=margin_atol
    end

    matrix_u = reshape(ctx.u, :, 1)
    @test cdf(C, matrix_u) ≈ [c] atol=numerical_atol
    @test logcdf(C, matrix_u) ≈ log.([c]) atol=1e-3
    @test_throws ArgumentError cdf(C, zeros(d + 1))
    @test_throws ArgumentError cdf(C, zeros(d + 1, 1))
end

test_density_contract(C, ctx) =
    test_density_contract(Copulas.copula_measure_style(C), C, ctx)
test_density_contract(::Copulas.NonAbsolutelyContinuousMeasure, C, ctx) = nothing
function test_density_contract(::Copulas.AbsolutelyContinuousMeasure, C, ctx)
    Base.@nospecialize C ctx
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    matrix_pdf = pdf(C, reshape(ctx.u, :, 1))

    @test p >= 0
    @test pdf(C, fill(1e-5, length(C))) >= 0
    @test pdf(C, fill(0.5, length(C))) >= 0
    @test pdf(C, fill(1 - 1e-5, length(C))) >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    @test matrix_pdf == [p]
    @test logpdf(C, reshape(ctx.u, :, 1)) ≈ log.(matrix_pdf)
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
    @test_throws DimensionMismatch logpdf(C, zeros(length(C) + 1))
    @test_throws ArgumentError logpdf(C, zeros(length(C) + 1, 1))
end

@testset verbose=true "public distribution-evaluation contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        ctx = copula_contract_context(C, 10_000 + seed)
        test_progress("operations", "distribution", case.name, "cdf")
        test_distribution_contract(C, ctx, case.numerical_atol, case.margin_atol)
        test_progress("operations", "distribution", case.name, "density")
        test_density_contract(C, ctx)
    end
end

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

@testset "Fréchet bounds use their exact mass identities" begin
    u = [0.37, 0.68]
    @test cdf(MCopula{2}(), u) == minimum(u)
    @test cdf(WCopula{2}(), u) == max(sum(u) - 1, 0)
end
