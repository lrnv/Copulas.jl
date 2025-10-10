using InteractiveUtils
using Copulas, Distributions
using Random
using StableRNGs


@testitem "Checking LogCopula == GumbelCopula" begin
    # [GenericTests integration]: Probably too specific (equivalence between two constructors/types). Could be a targeted identity test, keep here.

    rng = StableRNG(1234)
    for θ in [1.0, Inf, 0.5, rand(rng, Uniform(1.0, 10.0))]
        try
            C1 = LogCopula(2, θ)
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

@testitem "Extreme Galambos density test" begin
    # [GenericTests integration]: No. This is a trivial smoke test to catch crashes at extreme params; keep as minimal targeted test.
    rand(GalambosCopula(2, 19.7), 400)
    rand(GalambosCopula(2, 210.0), 400)
    @test true
end