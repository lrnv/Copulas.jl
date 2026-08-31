# Correctness obligation: independent closed forms and published numerical
# anchors for copula families not sharing a broader mechanism-specific file.

@testset "Raftery dependence anchors" begin
    # Retain the high-dimensional regimes that exposed exponent overflow.
    @test Copulas.τ(RafteryCopula{25}(0.5)) ≈ 0.18523466942807426
    @test isfinite(Copulas.ρ(RafteryCopula{100}(0.5)))
end

function raftery_cdf_oracle(θ, u)
    d = length(u)
    u_ordered = sort(u)
    term1 = u_ordered[1]
    term2 = ((1 - θ) * (1 - d)) / (1 - θ - d) * prod(u)^(1 / (1 - θ))
    term3 = zero(float(θ))
    for i in 2:d
        prod_prev = prod(u_ordered[1:(i - 1)])
        term3 += ((θ * (1 - θ)) / ((1 - θ - i) * (2 - θ - i))) *
                 prod_prev^(1 / (1 - θ)) *
                 u_ordered[i]^((2 - θ - i) / (1 - θ))
    end
    return term1 + term2 - term3
end

function raftery_pdf_oracle(θ, u)
    d = length(u)
    u_ordered = sort(u)
    term1 = inv((1 - θ)^(d - 1) * (1 - θ - d))
    term2 = 1 - d - θ * u_ordered[d]^((1 - θ - d) / (1 - θ))
    term3 = prod(u)^(θ / (1 - θ))
    return term1 * term2 * term3
end

@testset "Raftery manual CDF/PDF oracles" begin
    # Independent formulas and anchors from PR #137.
    u3 = [0.1, 0.2, 0.3]
    @test raftery_cdf_oracle(0.5, u3) ≈ 0.08236 atol=1e-4
    @test raftery_cdf_oracle(0.5, u3) ≈ cdf(RafteryCopula{3}(0.5), u3)
    @test raftery_cdf_oracle(0.8, [0.1, 0.2]) ≈
          cdf(RafteryCopula{2}(0.8), [0.1, 0.2])
    @test raftery_pdf_oracle(0.5, u3) ≈ 1.99450 atol=1e-4
    @test raftery_pdf_oracle(0.5, u3) ≈ pdf(RafteryCopula{3}(0.5), u3)
    @test raftery_pdf_oracle(0.8, [0.1, 0.2]) ≈
          pdf(RafteryCopula{2}(0.8), [0.1, 0.2])
end

@testset "Plackett reference CDF/PDF values" begin
    u = 0.1:0.18:1
    v = 0.4:0.1:0.9
    cdf_above = [0.055377800527509735, 0.1743883734874062,
        0.3166277269195278, 0.48232275012183223, 0.6743113969874872,
        0.8999999999999999]
    cdf_below = [0.026208734813001233, 0.10561162651259381,
        0.23491134194308438, 0.4162573282722253, 0.6419254774317229, 0.9]
    pdf_above = [1.0592107420343486, 1.023290881054283,
        1.038466936984394, 1.1100773231007635, 1.2729591789643138,
        1.652892561983471]
    pdf_below = [0.8446203068160272, 1.023290881054283,
        1.0648914416282562, 0.9360170818943749, 0.7346611825055718,
        0.5540166204986149]
    # Low interior, high interior, and unit-margin boundary in both regimes.
    for i in (1, 4, 6)
        @test cdf(PlackettCopula{2}(2.0), [u[i], v[i]]) ≈ cdf_above[i]
        @test cdf(PlackettCopula{2}(0.5), [u[i], v[i]]) ≈ cdf_below[i]
        @test pdf(PlackettCopula{2}(2.0), [u[i], v[i]]) ≈ pdf_above[i]
        @test pdf(PlackettCopula{2}(0.5), [u[i], v[i]]) ≈ pdf_below[i]
    end
end

@testset "multivariate FGM reference CDF/PDF values" begin
    cases = (
        ([0.1, 0.2, 0.5, 0.4], [0.1, 0.2, 0.3], 0.0100776123, 1.308876232),
        ([0.3, 0.3, 0.3, 0.3], [0.5, 0.4, 0.3], 0.0830421321, 1.024),
    )
    for (parameters, u, expected_cdf, expected_pdf) in cases
        C = FGMCopula{length(u)}(parameters)
        @test cdf(C, u) ≈ expected_cdf atol=1e-4
        @test pdf(C, u) ≈ expected_pdf atol=1e-4
    end
end
