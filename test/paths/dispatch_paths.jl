@testset "representative dispatch paths" begin
    for (name, C) in pairs(PATH_CASES)
        @testset "$name" begin
            d = length(C)
            u = fill(0.6, d)
            @test 0 <= cdf(C, u) <= 1
            @test size(rand(StableRNG(51), C, 2)) == (d, 2)
            D = condition(C, 1, 0.4)
            @test 0 <= cdf(D, 0.6) <= 1
        end
    end
end
