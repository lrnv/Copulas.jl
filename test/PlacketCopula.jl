# @testitem "PlackettCopula - Fix behavior of cdf, pdf and constructor" begin

#     # Fix the bahavior ofc the constructor: 
#     @test isa(PlackettCopula(1), IndependentCopula)
#     @test isa(PlackettCopula(Inf),WCopula) # should work in any dimenisons if theta is smaller than the bound.
#     @test isa(PlackettCopula(0),MCopula)

#     # Fix a few values for cdf and pdf:
#     using Distributions
#     u = 0.1:0.18:1
#     v = 0.4:0.1:0.9 
#     l1 = [0.055377800527509735, 0.1743883734874062, 0.3166277269195278, 0.48232275012183223, 0.6743113969874872, 0.8999999999999999]
#     l2 = [0.026208734813001233,   0.10561162651259381, 0.23491134194308438, 0.4162573282722253, 0.6419254774317229, 0.9]
#     l3 = [1.0592107420343486, 1.023290881054283, 1.038466936984394, 1.1100773231007635, 1.2729591789643138, 1.652892561983471]
#     l4 = [0.8446203068160272, 1.023290881054283, 1.0648914416282562, 0.9360170818943749, 0.7346611825055718, 0.5540166204986149]
#     for i in 1:6
#         @test cdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l1[i]
#         @test cdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l2[i]
#         @test pdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l3[i]
#         @test pdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l4[i]
#     end
# end