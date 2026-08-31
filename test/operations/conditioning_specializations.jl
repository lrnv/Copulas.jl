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
            test_progress("equivalence", "distortion quantile", name)
            generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, 0.63)
            @test isapprox(quantile(D, 0.63), generic; atol=2e-8, rtol=2e-8)
        end
    end
    @test !isempty(seen)
end

@testset verbose=true "bivariate conditioning routes agree with CDF derivatives" begin
    seen = Set{Method}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        is_absolutely_continuous(C) || continue
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),Tuple{Int},Tuple{Float64},Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            test_progress("equivalence", "bivariate conditioning", case.name)
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
        prove_dispatch_route!(:conditioning, C, :cdf_derivative)
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
    ν = C.df
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
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        is_absolutely_continuous(C) || continue
        js = Tuple(1:(d - 1))
        values = ntuple(k -> 0.3 + 0.08k, d - 1)
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),typeof(js),typeof(values),Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            test_progress("equivalence", "multivariate conditioning", case.name)
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
        prove_dispatch_route!(:conditioning, C,
                              :normalized_cdf_derivative)
    end
    @test !isempty(seen)
end

@testset "atomic conditioning routes satisfy generalized inversion" begin
    seen = Set{Any}()
    for fixture in ROUTING_COPULA_FIXTURES
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
        prove_dispatch_route!(:conditioning, C,
                              :generalized_quantile_identity)
    end
    @test !isempty(seen)
end

@testset "joint conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Any}()
    conditioned = 0.41
    h = 2e-5
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        key = dispatch_route_key(:conditional_joint, C)
        key in seen && continue
        push!(seen, key)

        H = condition(C, (1,), (conditioned,))
        targets = collect(range(0.53, 0.71; length=d - 1))
        conditional_scale = [cdf(H.m[i], targets[i]) for i in 1:(d - 1)]
        if C isa Union{GaussianCopula,TCopula}
            J, I = [1], collect(2:d)
            Σcond = C.Σ[I, I] - C.Σ[I, J] * (C.Σ[J, J] \ C.Σ[J, I])
            σ = sqrt.(diag(Σcond))
            expected_R = Σcond ./ (σ * σ')
            @test H.C.Σ ≈ expected_R atol=2e-12 rtol=2e-12
        elseif C isa LiouvilleCopula
            @test H.C isa LiouvilleCopula{d - 1}
            @test H.C.α == ntuple(i -> C.α[i + 1], d - 1)
        else
            upper = vcat(conditioned + h, targets)
            lower = vcat(conditioned - h, targets)
            numerator = (cdf(C, upper) - cdf(C, lower)) / (2h)
            normalizer = (cdf(C, vcat(conditioned + h, ones(d - 1))) -
                          cdf(C, vcat(conditioned - h, ones(d - 1)))) / (2h)
            tolerance = is_absolutely_continuous(C) ? 5e-4 : 3e-3
            @test isapprox(cdf(H.C, conditional_scale), numerator / normalizer;
                           atol=tolerance, rtol=tolerance)
        end
        prove_dispatch_route!(:conditional_joint, C,
                              :normalized_joint_cdf_derivative)
    end
    @test !isempty(seen)
end

@testset "conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula{4}(2.0)
    xf = [0.3, 0.5, 0.4, 0.6]
    xb = big.(xf)

    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db.den isa BigFloat
    @test eltype(db.uⱼₛ) === BigFloat
    cdf_db = cdf(db, xb[2])
    @test cdf_db isa BigFloat
    @test Float64(cdf_db) ≈ cdf(df, xf[2]) atol=1e-9

    mb = condition(C, (1, 3), Tuple(xb[[1, 3]]))
    @test mb.C.den isa BigFloat
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
