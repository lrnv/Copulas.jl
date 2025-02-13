@testitem "Checking LogCopula == GumbelCopula" begin
    using InteractiveUtils
    using Copulas, Distributions
    using Random
    using StableRNGs

    rng = StableRNG(1234)
    for θ in [1.0, Inf, 0.5, rand(rng, Uniform(1.0, 10.0))]
        try
            C1 = LogCopula(θ)
            C2 = GumbelCopula(2, θ)
            data = rand(rng, C1, 10)

            for i in 1:10
                u = data[:,i]
                cdf_value_C1 = cdf(C1, u)
                cdf_value_C2 = cdf(C2, u)
                pdf_value_C1 = pdf(C1, u)
                pdf_value_C2 = pdf(C2, u)

                @test isapprox(cdf_value_C1, cdf_value_C2, atol=1e-6) || error("CDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, cdf_value_C1=$cdf_value_C1, cdf_value_C2=$cdf_value_C2")
                @test isapprox(pdf_value_C1, pdf_value_C2, atol=1e-6) || error("PDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, pdf_value_C2=$pdf_value_C1, pdf_value_C2=$pdf_value_C2")
            end
        catch e
            @test e isa ArgumentError
            println("Could not construct LogCopula with θ=$θ: ", e)
        end
    end
end