# Family-regression layer: targeted miscellaneous-family identities,
# quantile regressions, boundary cases, and previously reported bugs.

@testset "Extreme-value quantiles use bounded bisection" begin
    for C in (
        BC2Copula{2}(0.5, 0.3),
        BC2Copula{2}(0.5516353577049822, 0.33689370624999193),
        BC2Copula{2}(0.6, 0.8),
    )
        D = Copulas.ExtremeDist(C.tail)
        for p in (0.1, 0.5, 0.9)
            q = quantile(D, p)
            @test 0 <= q <= 1
            @test cdf(D, q) >= p
        end
    end
end

@testset "Survival subsetting and conditioning regressions" begin
    C3 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (3,))
    S13 = subsetdims(C3, (1, 3))
    Sref = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    u = [0.25, 0.7]
    @test cdf(S13, u) ≈ cdf(Sref, u)
    @test pdf(S13, u) ≈ pdf(Sref, u)

    # Reordering changes flip positions, not their original dimension labels.
    C13 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (1, 3))
    S31 = subsetdims(C13, (3, 1))
    S31ref = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (1, 2))
    @test cdf(S31, u) ≈ cdf(S31ref, u)
    @test pdf(S31, u) ≈ pdf(S31ref, u)

    # Conditioning must remap surviving flipped dimensions as tuple values,
    # rather than passing the tuple type to the SurvivalCopula constructor.
    C24 = SurvivalCopula{4}(ClaytonCopula{4}(2.0), (2, 4))
    C24cond = condition(C24, (1, 3), (0.25, 0.75))
    @test C24cond.C isa SurvivalCopula{2}
    @test 0.0 <= cdf(C24cond, [0.4, 0.6]) <= 1.0

end

@testset "Raftery dependence anchors" begin
    # Bivariate and ordinary multivariate formulas are covered against exact
    # identities in the equivalence layer. Retain only the high-dimensional
    # numerical regimes that originally exposed exponent overflow.
    @test Copulas.τ(RafteryCopula{25}(0.5)) ≈ 0.18523466942807426
    @test isfinite(Copulas.ρ(RafteryCopula{100}(0.5)))
end

@testset "Check against manual version - CDF" begin
    # https://github.com/lrnv/Copulas.jl/pull/137
    function raftery_cdf_oracle(θ, u)
        d = length(u)
        u_ordered = sort(u)
        term1 = u_ordered[1]
        term2 = ((1 - θ) * (1 -d)) / (1 - θ - d) * prod(u)^(1/(1 - θ))
        term3 = zero(float(θ))
        for i in 2:d
            prod_prev = prod(u_ordered[1:i-1])
            term3_part = ((θ * (1 - θ)) / ((1 - θ - i) * (2 - θ - i))) * prod_prev^(1/(1 - θ)) * u_ordered[i]^((2 - θ - i) / (1 - θ))
            term3 += term3_part
        end
        return term1 + term2 - term3
    end
    @test raftery_cdf_oracle(0.5, [0.1,0.2,0.3]) ≈ 0.08236 atol=1e-4
    @test raftery_cdf_oracle(0.5, [0.1,0.2,0.3]) ≈ cdf(RafteryCopula{3}(0.5), [0.1,0.2,0.3])
    @test raftery_cdf_oracle(0.8, [0.1,0.2]) ≈ cdf(RafteryCopula{2}(0.8), [0.1,0.2])
end

@testset "Check against manual version - PDF" begin
    # https://github.com/lrnv/Copulas.jl/pull/137
    function raftery_pdf_oracle(θ, u)
        d = length(u)
        u_ordered = sort(u)
        term1 = (1/(((1-θ)^(d-1))*(1-θ-d)))
        term2 = (1-d-θ*(u_ordered[d])^((1-θ-d)/(1-θ)))
        term3 = (prod(u))^((θ)/(1-θ))
        return term1*term2*term3
    end
    @test raftery_pdf_oracle(0.5, [0.1,0.2,0.3]) ≈ 1.99450 atol=1e-4
    @test raftery_pdf_oracle(0.5, [0.1,0.2,0.3]) ≈ pdf(RafteryCopula{3}(0.5), [0.1,0.2,0.3])
    @test raftery_pdf_oracle(0.8, [0.1,0.2]) ≈ pdf(RafteryCopula{2}(0.8), [0.1,0.2])
end


@testset "PlackettCopula reference CDF and PDF values" begin
    # Fix a few values for cdf and pdf:
    u = 0.1:0.18:1
    v = 0.4:0.1:0.9 
    l1 = [0.055377800527509735, 0.1743883734874062, 0.3166277269195278, 0.48232275012183223, 0.6743113969874872, 0.8999999999999999]
    l2 = [0.026208734813001233,   0.10561162651259381, 0.23491134194308438, 0.4162573282722253, 0.6419254774317229, 0.9]
    l3 = [1.0592107420343486, 1.023290881054283, 1.038466936984394, 1.1100773231007635, 1.2729591789643138, 1.652892561983471]
    l4 = [0.8446203068160272, 1.023290881054283, 1.0648914416282562, 0.9360170818943749, 0.7346611825055718, 0.5540166204986149]
    # Low interior, high interior, and unit-margin boundary for both parameter
    # regimes; intermediate table rows use the same closed-form path.
    for i in (1, 4, 6)
        @test cdf(PlackettCopula{2}(2.0), [u[i], v[i]]) ≈ l1[i]
        @test cdf(PlackettCopula{2}(0.5), [u[i], v[i]]) ≈ l2[i]
        @test pdf(PlackettCopula{2}(2.0), [u[i], v[i]]) ≈ l3[i]
        @test pdf(PlackettCopula{2}(0.5), [u[i], v[i]]) ≈ l4[i]
    end
end

@testset "FGMCopula reference CDF and PDF values" begin
    cdf_exs = [
        ([0.1,0.2,0.5,0.4], [0.1, 0.2, 0.3], (0.0100776123, 1e-4), (1.308876232, 1e-4)),
        ([0.3,0.3,0.3,0.3], [0.5, 0.4, 0.3], (0.0830421321, 1e-4), (1.024, 1e-4)),
    ]
    
    for (par, u, (ctruth, ctol), (ptruth, ptol)) in cdf_exs
        copula = FGMCopula{length(u)}(par)
        @test cdf(copula, u) ≈ ctruth atol=ctol
        @test pdf(copula, u) ≈ ptruth atol=ptol
    end
end
