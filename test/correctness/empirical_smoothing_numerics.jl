@testset "stable empirical smoothing kernels" begin
    for T in (Float32, Float64), n in (10, 2000), p in (0.01, 0.5, 0.99)
        u = T(p)
        probabilities = Copulas._bernvec_n(u, n)
        reference = pdf.(Binomial(n, Float64(u)), 0:n)
        @test eltype(probabilities) === T
        @test sum(probabilities) ≈ one(T) rtol=20eps(T)
        # The independent Binomial oracle uses log-factorials; allow their
        # cancellation error at large n as well as the output precision.
        @test probabilities ≈ reference rtol=max(50eps(T), 10n * eps(Float64))
    end

    # Every empirical beta copula has exactly uniform margins, even when
    # the sample size makes an endpoint-started recurrence underflow.
    n = 2000
    data = vcat(reshape(1:n, 1, :), reshape(1:n, 1, :))
    beta = BetaCopula(data)
    for T in (Float32, Float64), p in (0.01, 0.5, 0.99)
        u = T(p)
        @test cdf(beta, [u, one(T)]) ≈ u rtol=50eps(T)
        @test cdf(beta, [one(T), u]) ≈ u rtol=50eps(T)
    end
    # Independent reference using the defining beta-kernel mixture.
    point = [0.4, 0.6]
    reference_cdf = sum(cdf(Beta(r, n + 1 - r), point[1]) *
                        cdf(Beta(r, n + 1 - r), point[2]) for r in 1:n) / n
    @test cdf(beta, point) ≈ reference_cdf rtol=1e-11
    @test isfinite(logpdf(beta, [0.01, 0.99]))
    @test logpdf(beta, [0.01, 0.99]) < -1000

    # For n=2 with matching ranks the density is the explicit polynomial
    # 2((1-u)(1-v)+uv); its two opposite corners have zero density.
    small = BetaCopula([1 2; 1 2])
    @test logpdf(small, [0.0, 1.0]) == -Inf
    @test logpdf(small, [1.0, 0.0]) == -Inf
    @test logpdf(small, [0.25, 0.75]) ≈ log(0.75)

    # Uniform Bernstein weights represent independence exactly. Unequal
    # degrees expose underflow without allocating a large square grid.
    bernstein = BernsteinCopula{2}((2000, 1), fill(1 / 2000, 2000, 1))
    for point in ([0.5, 1.0], [1.0, 0.5], [0.01, 0.7], [0.99, 0.3])
        @test cdf(bernstein, point) ≈ prod(point) rtol=1e-11
        @test logpdf(bernstein, point) ≈ 0 atol=1e-10
    end
    concentrated = BernsteinCopula{2}((2, 2), [0.5 0.0; 0.0 0.5])
    @test logpdf(concentrated, [0.0, 1.0]) == -Inf
    @test logpdf(concentrated, [0.25, 0.75]) ≈ log(0.75)

    # Compare tiny positive densities to a direct high-precision mixture,
    # independently of the log-space accumulation used by production code.
    rare_n = 400
    rare_point = [0.001, 0.999]
    rare_data = vcat(reshape(1:rare_n, 1, :), reshape(1:rare_n, 1, :))
    rare_beta = BetaCopula(rare_data)
    rare_bernstein = BernsteinCopula{2}(
        (rare_n, rare_n), Matrix(Diagonal(fill(1 / rare_n, rare_n))))
    reference_logdensity = setprecision(256) do
        density = sum(prod(pdf(Beta(BigFloat(r), BigFloat(rare_n + 1 - r)),
                               BigFloat(u)) for u in rare_point) for r in 1:rare_n) / rare_n
        Float64(log(density))
    end
    @test reference_logdensity < log(floatmin(Float64))
    @test logpdf(rare_beta, rare_point) ≈ reference_logdensity rtol=1e-12
    @test logpdf(rare_bernstein, rare_point) ≈ reference_logdensity rtol=1e-12

    # The recurrence remains differentiable for likelihood/conditioning
    # consumers, including when its mode changes at u=1/2.
    @test ForwardDiff.derivative(u -> cdf(beta, [u, one(u)]), 0.5) ≈ 1 rtol=1e-10
    @test ForwardDiff.derivative(u -> cdf(bernstein, [u, one(u)]), 0.5) ≈ 1 rtol=1e-10
    @test ForwardDiff.derivative(u -> logpdf(small, [u, 0.75]), 0.25) ≈ 4 / 3
end
