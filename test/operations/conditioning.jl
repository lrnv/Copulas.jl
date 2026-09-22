# Complete operation proof for marginal distortions and conditional copulas:
# public contracts, independent identities, specialization equivalence,
# family regressions, and route coverage are colocated here.

function test_conditioning_contract(C, u)
    Base.@nospecialize C u

    d = length(C)
    d > 2 && !is_absolutely_continuous(C) && return
    if d == 2
        scalar = condition(C, 1, u[1])
        tupled = condition(C, (1,), (u[1],))
        @test scalar isa Distributions.UnivariateDistribution
        @test cdf(scalar, u[2]) ≈ cdf(tupled, u[2])
    end
    if d > 2
        joint = condition(C, 1, u[1])
        @test length(joint) == d - 1
        @test 0 <= cdf(joint, u[2:end]) <= 1
    end
    if d > 3
        js = Tuple(1:(d - 2))
        joint = condition(C, js, Tuple(u[1:(d - 2)]))
        @test length(joint) == 2
        @test 0 <= cdf(joint, u[(d - 1):d]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(u[1:(d - 1)])
    D = condition(C, js, values)
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    q = quantile(D, 0.5)

    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test issorted(vals)
    @test logcdf(D, 0.5) ≈ log(cdf(D, 0.5))

    if is_absolutely_continuous(C)
        densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
        @test all(x -> x >= 0, densities)
        density = pdf(D, 0.5)
        @test iszero(density) ? logpdf(D, 0.5) == -Inf :
              logpdf(D, 0.5) ≈ log(density)
    end

    @test all(x -> 0 <= x <= 1, rand(StableRNG(73), D, 3))
    @test 0 <= q <= 1
    is_absolutely_continuous(C) &&
        @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end

function conditional_distribution(fixture)
    Base.@nospecialize fixture
    C = fixture.copula
    d = length(C)
    js = Tuple(1:(d - 1))
    values = ntuple(_ -> 0.4, d - 1)
    return condition(C, js, values)
end

function conditional_route_key(D)
    Base.@nospecialize D
    DT = typeof(D)
    return (
        which(Distributions.cdf, Tuple{DT,Float64}),
        which(Distributions.logcdf, Tuple{DT,Float64}),
        which(Distributions.logpdf, Tuple{DT,Float64}),
        which(Distributions.quantile, Tuple{DT,Float64}),
    )
end

const CONDITIONAL_DISTRIBUTION_CANDIDATES = [
    (fixture.case.name, conditional_distribution(fixture))
    for fixture in COPULA_FIXTURES
    if length(fixture.copula) == 2 ||
       is_absolutely_continuous(fixture.copula)
]

const CONDITIONAL_DISTRIBUTION_CASES = unique(
    case -> conditional_route_key(last(case)),
    CONDITIONAL_DISTRIBUTION_CANDIDATES,
)

conditional_measure_style(D::Copulas.Distortion) =
    Copulas.distortion_measure_style(D)
conditional_measure_style(::Distributions.UnivariateDistribution) =
    Copulas.AbsolutelyContinuousMeasure()

@testset "public conditioning contract" begin
    @testset "$(fixture.case.name)" for fixture in COPULA_FIXTURES
        test_conditioning_contract(
            fixture.copula,
            copula_contract_point(fixture.copula),
        )
    end
end


# Conditioning-operation equivalence proofs for optimized distortions and joint laws.

@testset verbose=true "all distortion quantile specializations agree with generic inversion" begin
    generic_method = which(quantile, Tuple{Copulas.Distortion,Real})
    seen = Set{Method}()
    for (name, D) in CONDITIONAL_DISTRIBUTION_CASES
        conditional_measure_style(D) isa Copulas.AbsolutelyContinuousMeasure ||
            continue
        D isa Copulas.Distortion || continue
        method = which(quantile, Tuple{typeof(D),Float64})
        method === generic_method && continue
        method in seen && continue
        push!(seen, method)
        @testset "$name" begin
            generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, 0.63)
            @test isapprox(quantile(D, 0.63), generic; atol=2e-8, rtol=2e-8)
        end
    end
    @test !isempty(seen)
end

@testset verbose=true "bivariate conditioning routes agree with CDF derivatives" begin
    seen = Set{Method}()
    for fixture in COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        is_absolutely_continuous(C) || continue
        method = which(Copulas.distortion,
            Tuple{typeof(C),Tuple{Int},Tuple{Float64},Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            conditioned, target = 0.41, 0.63
            D = condition(C, 1, conditioned)
            if D isa Copulas.LiouvilleDistortion
                x = quantile(D.margin, 1 - target)
                expected_cdf = ccdf(D.conditional_margin, x)
                expected_pdf = pdf(D.conditional_margin, x) / pdf(D.margin, x)
            elseif C isa GaussianCopula
                ρ = C.Σ[1, 2]
                zⱼ = quantile(Normal(), conditioned)
                zᵢ = quantile(Normal(), target)
                z = (zᵢ - ρ * zⱼ) / sqrt(1 - ρ^2)
                expected_cdf = cdf(Normal(), z)
                expected_pdf = pdf(Normal(), z) / (sqrt(1 - ρ^2) * pdf(Normal(), zᵢ))
            elseif C isa TCopula
                ν = C.ν
                ρ = C.Σ[1, 2]
                zⱼ = quantile(TDist(ν), conditioned)
                zᵢ = quantile(TDist(ν), target)
                σ = sqrt((ν + zⱼ^2) * (1 - ρ^2) / (ν + 1))
                z = (zᵢ - ρ * zⱼ) / σ
                expected_cdf = cdf(TDist(ν + 1), z)
                expected_pdf = pdf(TDist(ν + 1), z) /
                               (σ * pdf(TDist(ν), zᵢ))
            else
                h = 2e-5
                expected_cdf = (cdf(C, [conditioned + h, target]) -
                                cdf(C, [conditioned - h, target])) / (2h)
                expected_pdf = (
                    cdf(C, [conditioned + h, target + h]) -
                    cdf(C, [conditioned + h, target - h]) -
                    cdf(C, [conditioned - h, target + h]) +
                    cdf(C, [conditioned - h, target - h])
                ) / (4h^2)
            end
            @test isapprox(cdf(D, target), expected_cdf;
                           atol=3e-5, rtol=3e-5)
            @test isapprox(pdf(D, target), expected_pdf;
                           atol=3e-4, rtol=3e-4)
        end
    end
    @test !isempty(seen)
end

function _finite_conditional_cdf(C, js, values, target_index, target; h=2e-4)
    Base.@nospecialize C js values
    d = length(C)
    function mixed_at(target_value)
        total = 0.0
        for corner in Iterators.product(ntuple(_ -> (-1, 1), length(js))...)
            point = ones(d)
            point[target_index] = target_value
            for k in eachindex(js)
                point[js[k]] = values[k] + corner[k] * h
            end
            total += prod(corner) * cdf(C, point)
        end
        return total / (2h)^length(js)
    end
    return mixed_at(target) / mixed_at(1.0)
end

function _elliptical_conditional_cdf(C::GaussianCopula, js, values,
                                     target_index, target)
    J = collect(js)
    zJ = quantile.(Normal(), collect(values))
    β = C.Σ[J, J] \ C.Σ[J, target_index]
    μ = dot(C.Σ[target_index, J], C.Σ[J, J] \ zJ)
    σ² = 1 - dot(C.Σ[target_index, J], β)
    return cdf(Normal(), (quantile(Normal(), target) - μ) / sqrt(σ²))
end

function _elliptical_conditional_cdf(C::TCopula, js, values,
                                     target_index, target)
    J = collect(js)
    ν = C.ν
    zJ = quantile.(TDist(ν), collect(values))
    solved = C.Σ[J, J] \ zJ
    β = C.Σ[J, J] \ C.Σ[J, target_index]
    μ = dot(C.Σ[target_index, J], solved)
    σ0² = 1 - dot(C.Σ[target_index, J], β)
    δ = dot(zJ, solved)
    νp = ν + length(J)
    σ = sqrt(σ0² * (ν + δ) / νp)
    return cdf(TDist(νp), (quantile(TDist(ν), target) - μ) / σ)
end

@testset verbose=true "multivariate conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Method}()
    for fixture in COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        is_absolutely_continuous(C) || continue
        js = Tuple(1:(d - 1))
        values = ntuple(k -> 0.3 + 0.08k, d - 1)
        method = which(Copulas.distortion,
            Tuple{typeof(C),typeof(js),typeof(values),Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            target_index = d
            target = 0.63
            D = condition(C, js, values)
            expected = if C isa Union{GaussianCopula,TCopula}
                _elliptical_conditional_cdf(C, js, values, target_index, target)
            elseif D isa Copulas.LiouvilleDistortion
                x = quantile(D.margin, 1 - target)
                ccdf(D.conditional_margin, x)
            else
                _finite_conditional_cdf(C, js, values, target_index, target)
            end
            @test isapprox(cdf(D, target), expected; atol=2e-3, rtol=2e-3)
        end
    end
    @test !isempty(seen)
end

@testset "atomic conditioning routes satisfy generalized inversion" begin
    seen = Set{Any}()
    for fixture in COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        is_absolutely_continuous(C) && continue
        # Point conditioning is not canonically defined away from the finite
        # support of an empirical copula. Its generic method is exercised and
        # proved by the Raftery representative below.
        C isa EmpiricalCopula && continue
        key = dispatch_route_key(:conditioning, C)
        key in seen && continue
        push!(seen, key)
        d = length(C)
        D = condition(C, Tuple(1:(d - 1)), ntuple(_ -> 0.4, d - 1))
        @testset "$(case.name)" begin
            for p in (0.2, 0.6, 0.85)
                q = quantile(D, p)
                @test cdf(D, q) >= p - 1e-10
            end
        end
    end
    @test !isempty(seen)
end

@testset "mixed conditional laws use generalized quantiles" begin
    # A bijective Rosenblatt identity is intentionally not asserted in the
    # presence of atoms.
    C = MOCopula{2}(0.2, 0.3, 0.4)
    D = condition(C, 1, 0.4)
    probabilities = collect(0.05:0.05:0.95)
    quantiles = quantile.(Ref(D), probabilities)
    @test issorted(quantiles)
    @test any(iszero, diff(quantiles))
    for (p, q) in zip(probabilities, quantiles)
        @test cdf(D, q) >= p - 1e-10
    end
end

@testset "joint conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Any}()
    conditioned = 0.41
    h = 2e-5
    for fixture in COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        key = dispatch_route_key(:conditional_joint, C)
        key in seen && continue
        push!(seen, key)

        H = condition(C, (1,), (conditioned,))
        targets = collect(range(0.53, 0.71; length=d - 1))
        conditional_copula = H isa SklarDist ? H.C : H
        conditional_scale = H isa SklarDist ?
                            [cdf(H.m[i], targets[i]) for i in 1:(d - 1)] :
                            targets
        if C isa Union{GaussianCopula,TCopula}
            J, I = [1], collect(2:d)
            Σcond = C.Σ[I, I] - C.Σ[I, J] * (C.Σ[J, J] \ C.Σ[J, I])
            σ = sqrt.(diag(Σcond))
            expected_R = Σcond ./ (σ * σ')
            @test conditional_copula.Σ ≈ expected_R atol=2e-12 rtol=2e-12
        elseif C isa LiouvilleCopula
            @test conditional_copula isa LiouvilleCopula{d - 1}
            @test conditional_copula.α == ntuple(i -> C.α[i + 1], d - 1)
        else
            upper = vcat(conditioned + h, targets)
            lower = vcat(conditioned - h, targets)
            numerator = (cdf(C, upper) - cdf(C, lower)) / (2h)
            normalizer = (cdf(C, vcat(conditioned + h, ones(d - 1))) -
                          cdf(C, vcat(conditioned - h, ones(d - 1)))) / (2h)
            tolerance = is_absolutely_continuous(C) ? 5e-4 : 3e-3
            @test isapprox(cdf(conditional_copula, conditional_scale), numerator / normalizer;
                           atol=tolerance, rtol=tolerance)
        end
    end
    @test !isempty(seen)
end

@testset "conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula{4}(2.0)
    xf = [0.3, 0.5, 0.4, 0.6]
    xb = big.(xf)

    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db isa Copulas.ArchimedeanDistortion
    @test db.sJ isa BigFloat
    @test db.den isa BigFloat
    cdf_db = cdf(db, xb[2])
    @test cdf_db isa BigFloat
    @test Float64(cdf_db) ≈ cdf(df, xf[2]) atol=1e-9

    mb = condition(C, (1, 3), Tuple(xb[[1, 3]]))
    @test mb isa SklarDist
    @test mb.C isa ArchimedeanCopula{2}
    @test mb.C.G isa Copulas.TiltedGenerator
    @test mb.C.G.sJ isa BigFloat
    @test cdf(mb, xb[[2, 4]]) isa BigFloat

    C3 = ClaytonCopula{3}(2.0)
    @test condition(C3, 1, big"0.3") isa SklarDist
    X = SklarDist(C3, (Normal(), LogNormal(), Exponential()))
    big_conditioned = condition(X, (1,), (big"0.2",))
    float_conditioned = condition(X, (1,), (0.2,))
    @test big_conditioned isa SklarDist
    @test cdf(big_conditioned, [0.3, 0.5]) ≈
          cdf(float_conditioned, [0.3, 0.5]) atol=1e-6
    @test condition(ClaytonCopula{3}(2.0), (1, 2), (0.3f0, 0.4f0)) isa
          Copulas.Distortion
end



# Conditioning-operation equivalence: conditional-distribution and distortion
# fast paths are checked against inversion identities, generic conditionals,
# log-scale definitions, or independent Gaussian conditioning algebra.

@testset "Gaussian distortion log-scale formulas" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))
    N = Normal()
    for u in (1e-12, 0.2, 0.5, 0.8)
        q = quantile(N, u)
        z = (q - D.μz) / D.σz
        reference = logpdf(N, z) - log(abs(D.σz)) - logpdf(N, q)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 1e-13
        @test logpdf(D, u) ≈ reference atol = 1e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
    @test logpdf(D, -0.1) == -Inf
end

@testset "Student distortion logcdf" begin
    D = condition(TCopula{2}(4, [1.0 0.5; 0.5 1.0]), (1,), (0.3,))
    @test D.ν == 4
    @test D.νp == 5
    for u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "Student distortion preserves real degrees of freedom" begin
    C = TCopula{2}(4.5, [1.0 0.5; 0.5 1.0])
    D = condition(C, (1,), (0.3,))
    @test D.ν == 4.5
    @test D.νp == 5.5
    @test quantile(D, cdf(D, 0.37)) ≈ 0.37 atol=2e-12
end

@testset "Elliptical conditioning shares matrix factorizations" begin
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    for C in (GaussianCopula{3}(Σ), TCopula{3}(4, Σ))
        conditioned = condition(C, (1,), (0.35,))
        @test length(conditioned.m) == 2
        for (k, i) in enumerate((2, 3)), u in (0.2, 0.7)
            reference = Copulas.distortion(C, (1,), (0.35,), i)
            @test cdf(conditioned.m[k], u) ≈ cdf(reference, u) atol = 2e-12
        end
    end
end

@testset "Distorted distribution logcdf" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))(Logistic())
    @test D isa Copulas.DistortedDist
    for x in (-8.0, -0.5, 1.0)
        @test logcdf(D, x) ≈ logcdf(D.D, cdf(D.X, x)) atol = 2e-13
    end
end

@testset "Archimedean distortion logcdf" begin
    distortions = (
        condition(ClaytonCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(FrankCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(GumbelCopula{3}(2.0), (1, 2), (0.3, 0.6)),
    )
    for D in distortions, u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 3e-12
    end
    @test all(logcdf(D, 0.0) == -Inf for D in distortions)
    @test all(logcdf(D, 1.0) == 0.0 for D in distortions)
end

@testset "Flip distortion logcdf" begin
    S = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    D = condition(S, (1,), (0.3,))
    @test D isa Copulas.FlipDistortion
    for u in (0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-12
    end
    u = 1e-12
    @test logcdf(D, u) ≈ LogExpFunctions.log1mexp(logcdf(D.base, 1 - u)) atol = 2e-12
    @test isfinite(logcdf(D, u))
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "FGM distortion log-scale formulas" begin
    for θ in (-0.8, 0.8), uⱼ in (0.2, 0.7)
        D = condition(FGMCopula{2}(θ), (1,), (uⱼ,))
        for u in (1e-12, 0.2, 0.5, 0.8)
            @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-14
        end
        @test logcdf(D, 0.0) == -Inf
        @test logcdf(D, 1.0) == 0.0
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Generic ConditionalCopula density" begin
    C = GaussianCopula{3}([
        1.0 0.35 0.20
        0.35 1.0 0.25
        0.20 0.25 1.0
    ])
    js = (3,)
    ujs = (0.4,)
    generic = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, ujs)
    Cgeneric = FGMCopula{3}([0.1, 0.2, 0.3, 0.4])
    conditioned = condition(Cgeneric, js, ujs)
    @test conditioned.C isa Copulas.ConditionalCopula
    @test conditioned.m === conditioned.C.distortions
    @test conditioned.C.is == (1, 2)
    @test generic.logden == log(generic.den)
    specialized = Copulas.conditional_copula(C, js, ujs)

    for u in ([0.25, 0.35], [0.5, 0.5], [0.75, 0.65])
        @test isapprox(logpdf(generic, u), logpdf(specialized, u); atol=1e-8, rtol=1e-8)
        @test isapprox(pdf(generic, u), pdf(specialized, u); atol=1e-8, rtol=1e-8)
    end
    @test pdf(generic, [-0.1, 0.5]) == 0

    Cclayton = ClaytonCopula{3}(2.0)
    generic_big = @invoke Copulas.ConditionalCopula(
        Cclayton::Copulas.Copula{3},
        (3,),
        (big"0.4",),
    )
    value_big = logpdf(generic_big, BigFloat[0.35, 0.65])
    @test value_big isa BigFloat
    @test isfinite(value_big)
end


# Conditioning-operation proof: exercises the common univariate API once
# for every result reached through the public `condition` entry point. Most
# results are `Distortion`s, but families may legitimately return another
# `UnivariateDistribution`, such as BetaCopula's exact `MixtureModel`.
@testset "bivariate scalar conditioning contract" begin
    C = GaussianCopula{2}(0.4)
    @test @inferred(condition(C, 1, 0.4)) isa Copulas.GaussianDistortion
    for j in 1:2, uⱼ in (0.2f0, big(0.8))
        @test typeof(condition(C, j, uⱼ)) ==
              typeof(condition(C, (j,), (float(uⱼ),)))
    end
    @test_throws ArgumentError condition(C, 0, 0.4)
    @test_throws ArgumentError condition(C, 3, 0.4)
    @test_throws DomainError condition(C, 1, -0.1)
    @test_throws DomainError condition(C, 1, 1.1)
end

function test_distortion_contract(D)
    Base.@nospecialize D
    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test cdf(D, 0.0) == 0
    @test cdf(D, 1.0) ≈ 1
    @test cdf(D, -0.2) == 0
    upper = cdf(D, 1.2)
    if D isa Copulas.Distortion
        @test upper == 1
    else
        # Distributions.MixtureModel may accumulate a few ulps above one when
        # all component CDFs are numerically saturated. BetaCopula deliberately
        # returns that native mixture, so require numerical rather than bitwise
        # unity only for external univariate-distribution implementations.
        @test upper ≈ 1
    end

    # Two separated interior points prove monotonicity while avoiding repeated
    # numerical conditioning kernels for every concrete implementation.
    grid = (0.25, 0.75)
    values = cdf.(Ref(D), grid)
    @test issorted(values)
    @test all(x -> 0 <= x <= 1, values)
    @test all(u -> logcdf(D, u) ≈ log(cdf(D, u)), grid)

    # One generalized inverse call per implementation exercises the route;
    # inverse shape/ordering is covered by the distribution-level contracts.
    probabilities = (0.5,)
    quantiles = quantile.(Ref(D), probabilities)
    @test issorted(quantiles)
    @test all(x -> 0 <= x <= 1, quantiles)
    for (p, q) in zip(probabilities, quantiles)
        @test cdf(D, q) >= p - 2e-8
    end

    samples = rand(StableRNG(501), D, 1)
    @test all(x -> 0 <= x <= 1, samples)

    conditional_measure_style(D) isa Copulas.AbsolutelyContinuousMeasure || return
    @test pdf(D, -0.2) == 0
    @test pdf(D, 1.2) == 0
    @test logpdf(D, -0.2) == -Inf
    @test logpdf(D, 1.2) == -Inf
    for u in grid
        density = pdf(D, u)
        @test density >= 0
        @test iszero(density) ? logpdf(D, u) == -Inf :
              logpdf(D, u) ≈ log(density)
    end
end

@testset "conditional distributions satisfy the public contract" begin
    operations = (
        cdf=Distributions.cdf, logcdf=Distributions.logcdf,
        logpdf=Distributions.logpdf, quantile=Distributions.quantile,
    )
    selected_routes = Dict(name => Set(which(f, Tuple{typeof(D),Float64})
        for (_, D) in CONDITIONAL_DISTRIBUTION_CASES)
        for (name, f) in pairs(operations))
    checked_routes = Dict(name => Set{Method}() for name in keys(operations))
    for (name, D) in CONDITIONAL_DISTRIBUTION_CASES
        @testset "$name ($(nameof(typeof(D))))" begin
            test_distortion_contract(D)
            for (operation, f) in pairs(operations)
                push!(checked_routes[operation],
                      which(f, Tuple{typeof(D),Float64}))
            end
        end
    end
    @test checked_routes == selected_routes

    # Every route reached by full conditioning of a bivariate or multivariate
    # bestiary entry must be represented by the univariate contract.
    reachable = Set(conditional_route_key(D)
                    for (_, D) in CONDITIONAL_DISTRIBUTION_CANDIDATES)
    represented = Set(conditional_route_key(D)
                      for (_, D) in CONDITIONAL_DISTRIBUTION_CASES)
    @test reachable == represented
end

@testset "distortion push-forwards preserve the marginal scale" begin
    D = condition(GaussianCopula{2}(0.4), 1, 0.35)
    X = Logistic(0.3, 1.2)
    Y = D(X)
    for x in (0.2,)
        @test cdf(Y, x) ≈ cdf(D, cdf(X, x))
        @test pdf(Y, x) ≈ pdf(D, cdf(X, x)) * pdf(X, x)
    end
    @test D(Normal(0.3, 1.2)) isa Normal
    @test Copulas.NoDistortion()(X) === X
end

@testset "atomic distortion generalized quantiles" begin
    for D in (condition(MCopula{2}(), 1, 0.4),
              condition(WCopula{2}(), 1, 0.4))
        atom = quantile(D, 0.5)
        @test cdf(D, prevfloat(atom)) == 0
        @test cdf(D, atom) == 1
        @test cdf(D, nextfloat(atom)) == 1
        @test pdf(D, atom) == 1
        @test pdf(D, prevfloat(atom)) == 0
        @test all(==(atom), rand(StableRNG(502), D, 4))
    end
end

@testset "elementary distortions respect unit support" begin
    for D in (Copulas.NoDistortion(), Copulas.MDistortion(0.4, Int8(2)),
              Copulas.WDistortion(0.4, Int8(2)))
        @test cdf(D, -0.2) == 0
        @test cdf(D, 1.2) == 1
        @test pdf(D, -0.2) == 0
        @test pdf(D, 1.2) == 0
        @test logpdf(D, -0.2) == -Inf
        @test logpdf(D, 1.2) == -Inf
    end
end

# Focused regressions retain implementation-sensitive assertions that are not
# implied by the operation-wide mathematical contracts above.
@testset "Extreme-value conditioning caches fixed transforms" begin
    DEV = condition(GalambosCopula{2}(2.5), (1,), (0.3,))
    @test DEV.negloguⱼ == -log(DEV.uⱼ)

    DAM = condition(ArchimaxCopula{2}(Copulas.FrankGenerator(0.8),
        Copulas.HuslerReissTail(0.6)), (1,), (0.3,))
    @test DAM.yⱼ == Copulas.ϕ⁻¹(DAM.gen, DAM.uⱼ)
    @test DAM.invderivⱼ == Copulas.ϕ⁻¹⁽¹⁾(DAM.gen, DAM.uⱼ)
end

@testset "Checkerboard multidimensional conditioning regression" begin
    C = CheckerboardCopula{3}(randn(rng, 3, 30); pseudo_values=false)
    D = Copulas.distortion(C, (1, 2), (0.3, 0.7), 3)
    @test D isa Copulas.HistogramBinDistortion
    @test all(0 .<= cdf.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
    @test all(pdf.(Ref(D), (0.2, 0.5, 0.8)) .>= 0)
    @test all(0 .<= quantile.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
end

@testset "Bernstein distortion bounded inversion regression" begin
    D = condition(BernsteinCopula{2}(GaussianCopula{2}(0.3); m=5),
                  (1,), (0.4,))
    @test D isa Copulas.BernsteinDistortion
    for p in (0.1, 0.5, 0.9)
        q = quantile(D, p)
        @test 0 <= q <= 1
        @test cdf(D, q) ≈ p atol=2e-12
    end
end


@testset "Gumbel Liouville conditioning without AlphaStable density" begin
    G = Copulas.GumbelGenerator(2.3)
    C = LiouvilleCopula(G, (1, 2))

    # The global Gumbel radial route is deliberately unchanged:
    # sampling can still use the AlphaStable frailty.
    @test Copulas.𝒲₋₁(G, 3) isa Copulas.WilliamsonFromFrailty

    D = condition(C, 1, 0.3)

    @test D isa Copulas.LiouvilleDistortion

    # Conditioning deliberately uses the generic Williamson inversion
    # instead of WilliamsonFromFrailty{AlphaStable}.
    @test D.margin isa Copulas.𝒲₋₁
    @test D.conditional_margin isa Copulas.𝒲₋₁

    for u in (0.2, 0.5, 0.8)
        value = cdf(D, u)
        density = pdf(D, u)

        @test isfinite(value)
        @test 0 <= value <= 1
        @test isfinite(density)
        @test density >= 0
        @test logpdf(D, u) ≈ log(density)
    end

    for p in (0.2, 0.5, 0.8)
        q = quantile(D, p)

        @test isfinite(q)
        @test 0 <= q <= 1
        @test cdf(D, q) ≈ p atol=1e-8 rtol=1e-8
    end
end

@testset "Gumbel Liouville fractional conditioning" begin
    G = Copulas.GumbelGenerator(2.3)
    C = LiouvilleCopula(G, (1.25, 1.75))

    D = condition(C, 1, 0.3)

    @test D isa Copulas.LiouvilleDistortion

    # α₁ = 1.25 -> generic integer Williamson inverse + Beta reduction.
    @test D.margin isa Copulas.WilliamsonBetaProduct

    # Non-integer tilt uses the conditional-radial representation.
    @test D.conditional_margin isa Copulas.LiouvilleConditionalRadial

    for p in (0.2, 0.5, 0.8)
        q = quantile(D, p)

        @test isfinite(q)
        @test 0 <= q <= 1
        @test cdf(D, q) ≈ p atol=2e-7 rtol=2e-7
    end
end

@testset "BB6 Kendall tau θ = 1 boundary is exact" begin
    for δ in (1.5, 2.0, 5.0, Inf)
        τbb6 = Copulas.τ(BB6Copula{2}(1.0, δ))
        expected = isinf(δ) ? 1.0 : 1 - 1 / δ

        @test τbb6 == expected
    end
end

@testset "Archimedean independence boundaries use identity distortion" begin
    cases = (
        JoeCopula{2}(1.0),
        BB6Copula{2}(1.0, 1.0),
        BB8Copula{2}(1.0, 0.4),
        BB10Copula{2}(2.0, 0.0),
    )

    for C in cases
        D = condition(C, 1, 0.4)

        @test D isa Copulas.NoDistortion
        @test cdf(D, 0.2) == 0.2
        @test cdf(D, 0.7) == 0.7
        @test quantile(D, 0.6) == 0.6
        @test logpdf(D, 0.5) == 0.0
    end
end

@testset "Plackett boundary distortions" begin
    D_W = condition(PlackettCopula{2}(0.0), 1, 0.4)
    D_I = condition(PlackettCopula{2}(1.0), 1, 0.4)
    D_M = condition(PlackettCopula{2}(Inf), 1, 0.4)

    @test D_W isa Copulas.WDistortion
    @test D_I isa Copulas.NoDistortion
    @test D_M isa Copulas.MDistortion

    # W: U₂ = 1 - U₁.
    @test quantile(D_W, 0.2) ≈ 0.6
    @test quantile(D_W, 0.8) ≈ 0.6

    # Independence: conditional law stays uniform.
    @test quantile(D_I, 0.7) == 0.7
    @test cdf(D_I, 0.7) == 0.7

    # M: U₂ = U₁.
    @test quantile(D_M, 0.2) ≈ 0.4
    @test quantile(D_M, 0.8) ≈ 0.4
end

# Interval conditioning: U_I | U_js ∈ ∏ [lo, hi], a coordinate with lo == hi
# being a point. The oracle is rejection sampling on the unconditioned copula.
function _rejection_cdf(U, keep, thresholds)
    kept = view(U, :, keep)
    return mean(all(kept[i, col] <= thresholds[i] for i in eachindex(thresholds))
                for col in axes(kept, 2))
end

@testset "interval conditioning reduces to point conditioning" begin
    for C in (GaussianCopula([1.0 0.6 0.3; 0.6 1.0 0.5; 0.3 0.5 1.0]),
              ClaytonCopula(3, 2.0), GumbelCopula(3, 1.7))
        point = condition(C, (2, 3), (0.3, 0.7))
        box = condition(C, (2, 3), (0.3, 0.7), (0.3, 0.7))
        for u in 0.05:0.15:0.95
            @test cdf(box, u) ≈ cdf(point, u) atol=1e-12
            @test logpdf(box, u) ≈ logpdf(point, u) atol=1e-12
            @test quantile(box, u) ≈ quantile(point, u) atol=1e-12
        end
        joint_point = condition(C, (3,), (0.7,))
        joint_box = condition(C, (3,), (0.7,), (0.7,))
        @test cdf(joint_box, [0.4, 0.6]) ≈ cdf(joint_point, [0.4, 0.6]) atol=1e-12
    end
end

@testset "interval conditioning matches rejection sampling" begin
    C = ClaytonCopula(3, 2.0)
    U = rand(StableRNG(505), C, 1_000_000)
    keep = U[3, :] .<= 0.1

    # U₁, U₂ | U₃ ∈ [0, 0.1]: a SklarDist of the conditional copula and distortions.
    joint = condition(C, (3,), (0.0,), (0.1,))
    @test joint isa SklarDist
    @test length(joint) == 2
    for a in 0.1:0.2:0.9, b in 0.1:0.2:0.9
        @test cdf(joint, [a, b]) ≈ _rejection_cdf(U, keep, (a, b)) atol=5e-3
    end

    # U₁ | U₃ ∈ [0, 0.1], with U₂ free through its full box.
    margin = condition(C, (2, 3), (0.0, 0.0), (1.0, 0.1))
    @test margin isa Distributions.UnivariateDistribution
    for t in 0.05:0.1:0.95
        @test cdf(margin, t) ≈ mean(view(U, 1, keep) .<= t) atol=5e-3
    end
    # The midpoint of the interval is not a substitute.
    midpoint = condition(C, (2, 3), (1.0, 0.05))
    @test maximum(abs(cdf(midpoint, t) - cdf(margin, t)) for t in 0.05:0.05:0.95) > 0.05

    # Mixed: U₁ | U₂ = 0.7, U₃ ∈ [0, 0.1], against the point conditional rejected on U₃.
    mixed = condition(C, (2, 3), (0.7, 0.0), (0.7, 0.1))
    V = rand(StableRNG(506), condition(C, (2,), (0.7,)), 1_000_000)
    keep_mixed = V[2, :] .<= 0.1
    for t in 0.1:0.2:0.9
        @test cdf(mixed, t) ≈ mean(view(V, 1, keep_mixed) .<= t) atol=2e-2
    end
    @test quadgk(t -> pdf(mixed, t), 0, 1)[1] ≈ 1 atol=1e-6
    @test cdf(mixed, quantile(mixed, 0.37)) ≈ 0.37 atol=1e-8

    # Gaussian: the denominator is the probability of the box.
    G = GaussianCopula([1.0 0.6 0.3; 0.6 1.0 0.5; 0.3 0.5 1.0])
    UG = rand(StableRNG(507), G, 1_000_000)
    keep_g = (UG[2, :] .<= 0.2) .& (UG[3, :] .<= 0.2)
    DG = condition(G, (2, 3), (0.0, 0.0), (0.2, 0.2))
    @test DG.den ≈ Copulas.measure(G, [0.0, 0.0, 0.0], [1.0, 0.2, 0.2]) atol=1e-12
    @test DG.den ≈ mean(keep_g) atol=5e-3
    for t in 0.1:0.2:0.9
        @test cdf(DG, t) ≈ mean(view(UG, 1, keep_g) .<= t) atol=5e-3
    end
end

@testset "interval conditioning on the full box is the margin" begin
    C = ClaytonCopula(3, 2.0)
    full = condition(C, (2, 3), (0.0, 0.0), (1.0, 1.0))
    for t in 0.05:0.1:0.95
        @test cdf(full, t) ≈ t atol=1e-12
    end
    C4 = GumbelCopula(4, 1.8)
    full4 = condition(C4, (2, 3), (0.0, 0.0), (1.0, 1.0))
    S = subsetdims(C4, (1, 4))
    for u in ([0.3, 0.6], [0.5, 0.5], [0.8, 0.2])
        @test cdf(full4, u) ≈ cdf(S, u) atol=1e-10
    end
end

@testset "interval conditioning samples the conditional law" begin
    C = ClaytonCopula(4, 1.5)
    box = condition(C, (3, 4), (0.0, 0.2), (0.3, 0.9))
    sample = rand(StableRNG(508), box, 50_000)
    U = rand(StableRNG(509), C, 1_000_000)
    keep = (U[3, :] .<= 0.3) .& (0.2 .<= U[4, :] .<= 0.9)
    @test size(sample) == (2, 50_000)
    @test all(x -> 0 <= x <= 1, sample)
    @test StatsBase.corkendall(sample')[1, 2] ≈
          StatsBase.corkendall(U[1:2, keep]')[1, 2] atol=1e-2
    @test cdf(box, [0.3, 0.6]) ≈ _rejection_cdf(U, keep, (0.3, 0.6)) atol=5e-3
    @test isfinite(logpdf(box, [0.3, 0.6]))
    # The joint density integrates to one over the unit square.
    @test hcubature(v -> pdf(box, v), [0.0, 0.0], [1.0, 1.0]; rtol=1e-4)[1] ≈ 1 atol=1e-3
end

@testset "interval conditioning validates its box" begin
    C = ClaytonCopula(3, 2.0)
    @test_throws DomainError condition(C, (3,), (0.2,), (0.1,))
    @test_throws DomainError condition(C, (3,), (-0.1,), (0.1,))
    @test_throws DomainError condition(C, (3,), (0.5,), (1.5,))
    @test_throws ArgumentError condition(C, (3, 3), (0.1, 0.2), (0.3, 0.4))
    @test_throws DimensionMismatch condition(C, (2, 3), (0.1,), (0.3, 0.4))
    # A box of zero copula probability: the comonotone copula puts no mass off the diagonal.
    @test_throws ArgumentError condition(MCopula(3), (2, 3), (0.0, 0.5), (0.2, 0.7))
    @test_throws DomainError condition(SklarDist(C, (Normal(), Normal(), Normal())), 3, 1.0, 0.0)
    # Vector and scalar forms normalise like the point form.
    @test cdf(condition(C, [2, 3], [0.0, 0.0], [1.0, 0.1]), 0.4) ≈
          cdf(condition(C, (2, 3), (0.0, 0.0), (1.0, 0.1)), 0.4)
    @test cdf(condition(C, 3, 0.0, 0.1), [0.4, 0.6]) ≈
          cdf(condition(C, (3,), (0.0,), (0.1,)), [0.4, 0.6])
end

@testset "interval conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula(3, 2.0)
    Df = condition(C, (2, 3), (0.0, 0.0), (1.0, 0.1))
    Db = condition(C, (2, 3), (big"0.0", big"0.0"), (big"1.0", big"0.1"))
    @test Db.den isa BigFloat
    @test eltype(Db.lo) === BigFloat
    @test cdf(Db, big"0.4") isa BigFloat
    @test Float64(cdf(Db, big"0.4")) ≈ cdf(Df, 0.4) atol=1e-12
    @test logpdf(Db, big"0.4") isa BigFloat
    # Integer bounds are promoted with the copula's type, not cast.
    @test cdf(condition(C, (2, 3), (0, 0), (1, 0.1)), 0.4) ≈ cdf(Df, 0.4)
end

@testset "conditioning on a discrete observation uses its latent interval" begin
    X = SklarDist(ClaytonCopula(2, 2.0), (Normal(), Poisson(3.0)))
    sample = rand(StableRNG(510), X, 1_000_000)
    keep = sample[2, :] .== 2
    D = condition(X, 2, 2)
    for t in -2:0.5:2
        @test cdf(D, t) ≈ mean(view(sample, 1, keep) .<= t) atol=5e-3
    end
    @test quadgk(t -> pdf(D, t), -8, 8)[1] ≈ 1 atol=1e-6
    # The endpoint F(2) is the wrong latent value.
    endpoint = Copulas.distortion(X.C, (2,), (cdf(Poisson(3.0), 2),), 1)(Normal())
    @test maximum(abs(cdf(endpoint, t) - cdf(D, t)) for t in -2:0.25:2) > 0.05
    # An observation of zero probability is a zero-probability event.
    @test_throws ArgumentError condition(X, 2, -1)
    # Original-scale intervals on a discrete margin include both endpoints.
    Y = SklarDist(ClaytonCopula(2, 2.0), (Poisson(3.0), Normal()))
    sample_y = rand(StableRNG(511), Y, 1_000_000)
    keep_y = 1 .<= sample_y[1, :] .<= 3
    DY = condition(Y, 1, 1, 3)
    for t in -2:0.5:2
        @test cdf(DY, t) ≈ mean(view(sample_y, 2, keep_y) .<= t) atol=5e-3
    end
    # A discrete free margin: the conditional pmf sums to one and cumulates to its cdf.
    DP = condition(Y, 2, 0.4)
    masses = [pdf(DP, k) for k in 0:40]
    @test sum(masses) ≈ 1 atol=1e-10
    @test cdf(DP, 3) ≈ sum(masses[1:4]) atol=1e-10
    @test pdf(DP, 2.5) == 0
    @test quantile(DP, cdf(DP, 3) - 1e-9) == 3
    # A continuous SklarDist follows the point path unchanged.
    Z = SklarDist(ClaytonCopula(2, 2.0), (Normal(), Normal()))
    @test condition(Z, 2, 0.4) isa Copulas.DistortedDist
    @test cdf(condition(Z, 2, 0.4, 0.4), 0.3) == cdf(condition(Z, 2, 0.4), 0.3)
    # A mixed observed set is one call: X₁ | X₂ = 2, X₃ = 0.3 under a 3-D model.
    W = SklarDist(ClaytonCopula(3, 2.0), (Normal(), Poisson(3.0), Normal()))
    DW = condition(W, (2, 3), (2, 0.3))
    Vw = rand(StableRNG(512), condition(W, 3, 0.3), 1_000_000)
    keep_w = Vw[2, :] .== 2
    for t in -1:0.5:1
        @test cdf(DW, t) ≈ mean(view(Vw, 1, keep_w) .<= t) atol=1e-2
    end
end

@testset "generic Gumbel partials are finite at a free coordinate of 1" begin
    # `log(-log(1))` is a NaN dual, and the generic AD path places every free
    # coordinate at 1, so the generic distortion of a 3-D Gumbel was NaN.
    C = GumbelCopula(3, 1.6)
    @test Copulas._partial_cdf(C, (2, 3), (1,), (0.4, 1.0), (0.7,)) ≈
          Copulas._partial_cdf(C, (2, 3), (1,), (0.4, prevfloat(1.0)), (0.7,)) atol=1e-8
    generic = Copulas.DistortionFromCop(C, (1,), (0.7,), 2)
    @test cdf(generic, 0.4) ≈ cdf(Copulas.distortion(C, (1,), (0.7,), 2), 0.4) atol=1e-10
    @test isfinite(ForwardDiff.derivative(t -> cdf(C, [0.7, 0.4, t]), 1.0))
    @test cdf(C, [1.0, 1.0, 1.0]) == 1
    B = BB3Copula(3, 1.5, 0.7)
    @test isfinite(Copulas._partial_cdf(B, (2, 3), (1,), (0.4, 1.0), (0.7,)))
    @test isfinite(cdf(condition(C, (2, 3), (0.0, 0.0), (1.0, 0.1)), 0.4))
end