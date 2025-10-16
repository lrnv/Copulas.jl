
@testset "Testing survival stuff" begin
    # [GenericTests integration]: Yes. Symmetry of survival transformations on pdf/cdf is generic; we can add survival invariance checks.
    Random.seed!(rng,123)
    C = ClaytonCopula(2,3.0) # bivariate clayton with theta = 3.0
    C90 = SurvivalCopula(C,(1,)) # flips the first dimension
    C270 = SurvivalCopula(C,(2,)) # flips only the second dimension. 
    C180 = SurvivalCopula(C,(1,2)) # flips both dimensions.

    u1,u2 = rand(rng,2)
    p = pdf(C,[u1,u2])
    @test pdf(C90,[1-u1,u2]) == p
    @test pdf(C270,[u1,1-u2]) == p
    @test pdf(C180,[1-u1,1-u2]) == p

end

@testset "RafteryCopula Constructor" begin
    # [GenericTests integration]: Partially. Constructor mapping to degenerate copulas (Independent/MCopula) could be generalized; keep argument errors here.
    for d in [2,3,4]
        @test isa(RafteryCopula(d,0.0), IndependentCopula)
        @test isa(RafteryCopula(d,1.0), MCopula)
    end
    @test_throws ArgumentError RafteryCopula(3,-1.5)
    @test_throws ArgumentError RafteryCopula(2, 2.6)
end

@testset "RafteryCopula CDF" begin
    # [GenericTests integration]: Maybe. The numeric values are specific regression checks; a lighter generic monotonicity/nonnegativity check exists.
    Random.seed!(rng,123)
    for d in [2, 3, 4]
        F = RafteryCopula(d, 0.5)
        cdf_value = cdf(F, rand(d))
        pdf_value = pdf(F,rand(d))
        @test cdf_value >= 0 && cdf_value <= 1
        @test pdf_value >= 0 
    end

    @test cdf(RafteryCopula(2, 0.8), [0.2, 0.5]) ≈ 0.199432 atol=1e-5
    @test cdf(RafteryCopula(2, 0.5), [0.3, 0.8]) ≈ 0.2817 atol=1e-5
    @test cdf(RafteryCopula(3, 0.5), [0.1, 0.2, 0.3]) ≈ 0.08236007 atol=1e-5
    @test cdf(RafteryCopula(3, 0.1), [0.4, 0.8, 0.2]) ≈ 0.08581997 atol=1e-5    

    @test pdf(RafteryCopula(2, 0.8), [0.2, 0.5]) ≈ 0.114055555 atol=1e-4
    @test pdf(RafteryCopula(2, 0.5), [0.3, 0.8]) ≈ 0.6325 atol=1e-4
    @test pdf(RafteryCopula(3, 0.5), [0.1, 0.2, 0.3]) ≈ 1.9945086 atol=1e-4
    @test pdf(RafteryCopula(3, 0.1), [0.4, 0.8, 0.2]) ≈ 0.939229 atol=1e-4
end

@testset "Check against manual version - CDF" begin
    # [GenericTests integration]: No. Manual formula replication is too bespoke; keep as targeted verification for this copula.
    # https://github.com/lrnv/Copulas.jl/pull/137
    function prueba_CDF(R::Vector{T}, u::Vector{T}) where T
        # Order the vector u
        θ = R[1]
        # println("param:", θ)
        d = round(Int, R[2])
        # println("dimension:", d)
        u_ordered = sort(u)
        # println("Sorted vector: ", u_ordered)
        
        term1 = u_ordered[1]
        # println("Term 1: ", term1)
        
        term2 = ((1 - θ) * (1 -d)) / (1 - θ - d) * prod(u)^(1/(1 - θ))
        # println("Term 2: ", term2)
        
        term3 = 0.0
        for i in 2:d # <<<<<<<<--------------- This should be 2:d and not 2:length(R) since length(R) is not the dimension. 
            prod_prev = prod(u_ordered[1:i-1])
            term3_part = ((θ * (1 - θ)) / ((1 - θ - i) * (2 - θ - i))) * prod_prev^(1/(1 - θ)) * u_ordered[i]^((2 - θ - i) / (1 - θ))
            # println("Term 3 (part $i): ", term3_part)
            term3 += term3_part
        end
        
        # Combine the terms to get the cumulative distribution function
        cdf_value = term1 + term2 - term3
        # println("Final CDF value: ", cdf_value)
        
        return cdf_value
    end
    @test prueba_CDF([0.5,3], [0.1,0.2,0.3]) ≈ 0.08236 atol=1e-4 # According to https://github.com/lrnv/Copulas.jl/pull/137#issuecomment-1953365273
    @test prueba_CDF([0.5,3], [0.1,0.2,0.3]) ≈ cdf(RafteryCopula(3,0.5), [0.1,0.2,0.3])
    @test prueba_CDF([0.8,2], [0.1,0.2]) ≈ cdf(RafteryCopula(2,0.8), [0.1,0.2])
    @test prueba_CDF([0.2,2], [0.8,0.2]) ≈ cdf(RafteryCopula(2,0.2), [0.8,0.2])
end

@testset "Check against manual version - PDF" begin
    # [GenericTests integration]: No. Same rationale as CDF manual check; keep here.
    # https://github.com/lrnv/Copulas.jl/pull/137
    function prueba_PDF(R::Vector{T}, u::Vector{T}) where T
        # Order the vector u
        θ = R[1]
        d = round(Int, R[2])
        u_ordered = sort(u)
        # println("Sorted vector: ", u_ordered)
        
        term1 = (1/(((1-θ)^(d-1))*(1-θ-d)))
        # println("Term 1: ", term1)
        
        term2 = (1-d-θ*(u_ordered[d])^((1-θ-d)/(1-θ)))
        # println("Term 2: ", term2)
        
        term3 = (prod(u))^((θ)/(1-θ))
        # println(term3)
        # Combine the terms to get the density distribution function
        pdf_value = term1*term2*term3
        # println("Final PDF value: ", pdf_value)
        
        return pdf_value
    end
    @test prueba_PDF([0.5,3], [0.1,0.2,0.3]) ≈ 1.99450 atol=1e-4 # According to https://github.com/lrnv/Copulas.jl/pull/137#issuecomment-1953375141
    @test prueba_PDF([0.5,3], [0.1,0.2,0.3]) ≈ pdf(RafteryCopula(3,0.5), [0.1,0.2,0.3])
    @test prueba_PDF([0.8,2], [0.1,0.2]) ≈ pdf(RafteryCopula(2,0.8), [0.1,0.2])
    @test prueba_PDF([0.2,2], [0.8,0.2]) ≈ pdf(RafteryCopula(2,0.2), [0.8,0.2])
end


@testset "PlackettCopula - Fix behavior of cdf, pdf and constructor" begin
    # [GenericTests integration]: Partially. Constructor edge cases can be made generic; the fixed value grids are regression tests, keep here.

    # Fix the bahavior ofc the constructor: 
    @test isa(PlackettCopula(1), IndependentCopula)
    @test isa(PlackettCopula(Inf),WCopula) # should work in any dimenisons if theta is smaller than the bound.
    @test isa(PlackettCopula(0),MCopula)

    # Fix a few values for cdf and pdf:
    u = 0.1:0.18:1
    v = 0.4:0.1:0.9 
    l1 = [0.055377800527509735, 0.1743883734874062, 0.3166277269195278, 0.48232275012183223, 0.6743113969874872, 0.8999999999999999]
    l2 = [0.026208734813001233,   0.10561162651259381, 0.23491134194308438, 0.4162573282722253, 0.6419254774317229, 0.9]
    l3 = [1.0592107420343486, 1.023290881054283, 1.038466936984394, 1.1100773231007635, 1.2729591789643138, 1.652892561983471]
    l4 = [0.8446203068160272, 1.023290881054283, 1.0648914416282562, 0.9360170818943749, 0.7346611825055718, 0.5540166204986149]
    for i in 1:6
        @test cdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l1[i]
        @test cdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l2[i]
        @test pdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l3[i]
        @test pdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l4[i]
    end
end

@testset "Fixing values of FGMCopula - cdf, pdf, constructor" begin
    # [GenericTests integration]: Partially. Constructor-to-independent is generic; the numeric regression grids for cdf/pdf should stay specific.

    @test isa(FGMCopula(2,0.0), IndependentCopula)
    Random.seed!(rng,123)

    cdf_exs = [
        ([0.1,0.2,0.5,0.4], [0.1, 0.2, 0.3], (0.0100776123, 1e-4), (1.308876232, 1e-4)),
        ([0.3,0.3,0.3,0.3], [0.5, 0.4, 0.3], (0.0830421321, 1e-4), (1.024, 1e-4)),
        (0.0,               [0.1, 0.1],      (0.010023, 1e-4),     (1, 1e-4)),
        (0.5,               [0.5, 0.4],      (0.2299999999, 1e-4),     (1, 1e-4)),
    ]
    
    for (par, u, (ctruth, ctol), (ptruth, ptol)) in cdf_exs
        copula = FGMCopula(length(u), par)
        @test cdf(copula, u) ≈ ctruth atol=ctol
        @test pdf(copula, u) ≈ ptruth atol=ptol
    end
end