@testitem "RafteryCopula Constructor" begin
    for d in [2,3,4]
        @test isa(RafteryCopula(d,0.0), IndependentCopula)
        @test isa(RafteryCopula(d,1.0), MCopula)
    end    
    @test_throws ArgumentError RafteryCopula(3,-1.5)
    @test_throws ArgumentError RafteryCopula(2, 2.6)
end

@testitem "RafteryCopula CDF" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
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

@testitem "RafteryCopula Sampling" begin
    using StableRNGs
    rng = StableRNG(123)
    n_samples = 100
    F = RafteryCopula(3,0.5)
    samples = rand(rng,F, n_samples)
    @test size(samples) == (3, n_samples)
end

@testitem "Check against manual version - CDF" begin
    # https://github.com/lrnv/Copulas.jl/pull/137
    using Distributions
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

@testitem "Check against manual version - PDF" begin
    # https://github.com/lrnv/Copulas.jl/pull/137
    using Distributions
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