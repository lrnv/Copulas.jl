# @testitem "test default parametrisation of every concrete copula types" begin
#     for T in subtypes(Copulas)
#         if isconcretetype(T)
#             rand(T(),100)
#             @test true
#         end
#     end
# end

# @testitem "test sampling of every concrete copula type" begin
#     for T in subtypes(Copulas)
#         if isconcretetype(T)
#             rand(T(),100)
#             @test true
#         end
#     end
# end


@testitem "small sampling test" begin
    # test constructed from https://github.com/lrnv/Copulas.jl/issues/35
    using Copulas, Distributions
    rand(GaussianCopula([1.0 0.5; 0.5 1.0]),1)
    rand(IndependentCopula(2),1)
    rand(MCopula(2),1)
    rand(WCopula(2),1)

    @test true
end


@testitem "test cdf for three copulas." begin
    # test constructed from https://github.com/lrnv/Copulas.jl/issues/35
    using Copulas, Distributions

    u = range(0, stop=1, length=100)

    for G in (
        GaussianCopula([1.0 0.5; 0.5 1.0]),
        IndependentCopula(2),
        MCopula(2),
        WCopula(2)
    )
        for uᵢ in u
            for vᵢ in u
                cdf(G,[uᵢ,vᵢ])
            end
        end
    end
    @test true
end

@testitem "test pdf for three copulas." begin
    # test constructed from https://github.com/lrnv/Copulas.jl/issues/35
    using Copulas, Distributions

    u = range(0, stop=1, length=100)

    for G in (
        GaussianCopula([1.0 0.5; 0.5 1.0]),
        IndependentCopula(2),
        # MCopula(2)
    )
        for uᵢ in u
            for vᵢ in u
                pdf(G,[uᵢ,vᵢ])
            end
        end
    end
    @test true
end