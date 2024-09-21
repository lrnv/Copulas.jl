# @testitem "Extreme frank density test" begin
#     using Distributions
#     F = FrankCopula(2, 60)
#     den = pdf(F,[0.70, 0.66])
#     @test true
# end