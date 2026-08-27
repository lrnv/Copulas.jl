# Mechanism-path layer: exercises one representative of each important generic
# or specialized sampling, conditioning, subsetting, and numerical dispatch path.
# Deterministic specializations are compared with `invoke`-selected fallbacks
# below. Generator and tail primitives use their independent mathematical
# references in `components/`; sampler-only and atomic paths use distributional
# identities in `statistical_paths.jl` instead of meaningless draw-by-draw tests.
_which(f, args...) = which(f, Tuple{typeof.(args)...})

function _dispatch_path(operation, C, case)
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        return _which(Copulas._cdf, C, u)
    elseif operation === :logpdf
        case.kind === :continuous || return nothing
        return _which(Distributions._logpdf, C, u)
    elseif operation === :sampling
        return _which(Distributions._rand!, StableRNG(51), C, zeros(d, 1))
    elseif operation === :conditioning
        js = Tuple(1:(d - 1))
        values = ntuple(_ -> 0.4, d - 1)
        return _which(Copulas.DistortionFromCop, C, js, values, d)
    elseif operation === :rosenblatt
        case.rosenblatt || return nothing
        return _which(Copulas.rosenblatt, C, reshape(u, :, 1))
    elseif operation === :subsetting
        dims = d == 2 ? (2, 1) : (1, d)
        return _which(Copulas.subsetdims, C, dims)
    end
    error("unknown dispatch operation $operation")
end

function _exercise_dispatch_path(operation, C)
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        @test 0 <= cdf(C, u) <= 1
    elseif operation === :logpdf
        @test !isnan(logpdf(C, u))
    elseif operation === :sampling
        @test size(rand(StableRNG(51), C, 2)) == (d, 2)
    elseif operation === :conditioning
        D = condition(C, Tuple(1:(d - 1)), ntuple(_ -> 0.4, d - 1))
        @test 0 <= cdf(D, 0.6) <= 1
    elseif operation === :rosenblatt
        @test size(rosenblatt(C, reshape(u, :, 1))) == (d, 1)
    elseif operation === :subsetting
        @test length(subsetdims(C, d == 2 ? (2, 1) : (1, d))) == 2
    end
end

@testset "one representative per copula dispatch mechanism" begin
    models = Tuple((case=case, copula=case.build()) for case in COPULA_CASES)
    for operation in (:cdf, :logpdf, :sampling, :conditioning, :rosenblatt, :subsetting)
        seen = Set{Any}()
        for (; case, copula) in models
            method = _dispatch_path(operation, copula, case)
            isnothing(method) && continue
            key = (method, length(copula) == 2 ? :bivariate : :multivariate)
            key in seen && continue
            push!(seen, key)
            @info "Testing dispatch mechanism" operation copula=case.name method
            _exercise_dispatch_path(operation, copula)
        end
        @test !isempty(seen)
    end
end

@testset "specialized FGM paths agree with the generic polynomial oracle" begin
    θ = 0.4
    generic = PolynomialOracleCopula(θ)
    specialized = FGMCopula{2}(θ)
    u = [0.37, 0.68]

    generic_integrated_cdf =
        invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, generic, u)
    @test cdf(specialized, u) ≈ generic_integrated_cdf atol=2e-5
    @test pdf(specialized, u) ≈ pdf(generic, u)
    @test Copulas.measure(specialized, [0.15, 0.25], [0.55, 0.65]) ≈
          Copulas.measure(generic, [0.15, 0.25], [0.55, 0.65])

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2])
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6

    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
    @test inverse_rosenblatt(specialized, rosenblatt(specialized, u)) ≈
          inverse_rosenblatt(generic, rosenblatt(generic, u)) atol=2e-6
    @test Copulas.ρ(specialized) ≈ Copulas.ρ(generic) atol=2e-5
    @test Copulas.β(specialized) ≈ Copulas.β(generic)
end

@testset "specialized Gumbel generator agrees with its generic oracle" begin
    θ = 1.5
    generic = PowerExponentialOracleGenerator(θ)
    specialized = Copulas.GumbelGenerator(θ)
    for t in (0.2, 0.7, 1.4)
        p = Copulas.ϕ(generic, t)
        @test Copulas.ϕ(specialized, t) ≈ p
        @test Copulas.ϕ⁻¹(specialized, p) ≈ Copulas.ϕ⁻¹(generic, p)
        @test Copulas.ϕ⁽¹⁾(specialized, t) ≈ Copulas.ϕ⁽¹⁾(generic, t)
        @test Copulas.ϕ⁽ᵏ⁾(specialized, 2, t) ≈
              Copulas.ϕ⁽ᵏ⁾(generic, 2, t)
        @test Copulas.ϕ⁻¹⁽¹⁾(specialized, p) ≈ Copulas.ϕ⁻¹⁽¹⁾(generic, p)
    end
end

@testset "specialized logistic tail agrees with its generic oracle" begin
    θ = 1.5
    generic_tail = LogisticOracleTail(θ)
    specialized_tail = Copulas.LogTail(θ)
    x = [0.4, 0.7]
    weight = Tuple(x ./ sum(x))
    @test Copulas.ℓ(specialized_tail, x) ≈ Copulas.ℓ(generic_tail, x)
    @test Copulas.A(specialized_tail, weight) ≈ Copulas.A(generic_tail, weight)
    for indices in ((), (1,), (2,), (1, 2))
        @test Copulas.ellpartial(specialized_tail, x, indices) ≈
              Copulas.ellpartial(generic_tail, x, indices) atol=2e-6
    end

    generic = ExtremeValueCopula{2}(generic_tail)
    specialized = LogCopula{2}(θ)
    u = [0.37, 0.68]
    @test cdf(specialized, u) ≈ cdf(generic, u)
    @test pdf(specialized, u) ≈ pdf(generic, u) atol=2e-6

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2]) atol=2e-6
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6
    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
end

@testset "closed-form distortion quantiles agree with the generic inverse" begin
    distortions = (
        condition(PlackettCopula{2}(2.0), 1, 0.4),
        condition(FrankCopula{2}(2.0), 1, 0.4),
        condition(GumbelCopula{2}(2.0), 1, 0.4),
        condition(InvGaussianCopula{2}(0.5), 1, 0.4),
        condition(GumbelBarnettCopula{2}(0.5), 1, 0.4),
    )
    for D in distortions, p in (0.2, 0.7)
        generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, p)
        @test quantile(D, p) ≈ generic atol=2e-8 rtol=2e-8
    end
end

@testset "specialized Rosenblatt implementations agree with the generic path" begin
    u = [0.2 0.7; 0.4 0.6; 0.8 0.3]
    for C in (
        ClaytonCopula{3}(1.5),
        GaussianCopula{3}([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
        TCopula{3}(5, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    )
        specialized = rosenblatt(C, u)
        generic = invoke(Copulas.rosenblatt,
            Tuple{Copulas.Copula{3},AbstractMatrix{<:Real}}, C, u)
        @test specialized ≈ generic atol=3e-10
        @test inverse_rosenblatt(C, specialized) ≈ u atol=3e-10
    end
end

@testset "EV analytic partials agree with the differentiable CDF path" begin
    f(z) = z[1]^2 * z[2]^3 + z[3]
    mixed_point = [0.4, 0.7, 1.1]
    expected12 = 6 * mixed_point[1] * mixed_point[2]^2
    @test Copulas._mixed_partial(f, mixed_point, (1, 2)) ≈ expected12
    @test Copulas._mixed_partial(f, Tuple(mixed_point), [1, 2]) ≈ expected12

    z = [0.31, 0.57, 0.73]
    for C in (LogCopula{3}(2.0), GalambosCopula{3}(0.7))
        analytic = Copulas._partial_cdf(C, (3,), (1, 2),
                                        (z[3],), (z[1], z[2]))
        differentiated = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                b -> cdf(C, [a, b, z[3]]), z[2]), z[1])
        @test analytic ≈ differentiated atol=1e-11 rtol=2e-8
    end

    # Numerical-kernel tails cannot accept dual numbers; their analytic STDF
    # partials must nevertheless power conditioning and Rosenblatt end to end.
    C = tEVCopula{3}(4.0, 0.2)
    D = condition(C, (1, 2), (0.31, 0.58))
    q = quantile(D, 0.6)
    @test cdf(D, q) ≈ 0.6 atol=2e-6 rtol=2e-6
    u = [0.21, 0.53, 0.74]
    @test inverse_rosenblatt(C, rosenblatt(C, u)) ≈ u atol=2e-6 rtol=2e-6
end

@testset "conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula{4}(2.0)
    xf = [0.3, 0.5, 0.4, 0.6]
    xb = big.(xf)

    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db.den isa BigFloat
    @test eltype(db.uⱼₛ) === BigFloat
    @test cdf(db, xb[2]) isa BigFloat
    @test Float64(cdf(db, xb[2])) ≈ cdf(df, xf[2]) atol=1e-9

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


@testset "generic numeric sampler buffers" begin
    C = ClaytonCopula{3}(1.0)
    storage = fill(Float32(NaN), 5, 2)
    buffer = @view storage[2:4, :]
    @test rand!(StableRNG(52), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    @test all(isnan, storage[[1, 5], :])
    @test_throws DimensionMismatch rand!(StableRNG(52), C, zeros(Float32, 2, 1))
end
