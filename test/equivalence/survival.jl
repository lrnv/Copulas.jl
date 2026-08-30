# Equivalence obligation: subsetting a SurvivalCopula preserves the parent
# margin and remaps flipped coordinates into the requested order.
@testset "Survival subsetting equivalence" begin
    u = [0.25, 0.7]

    C3 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (3,))
    subset = subsetdims(C3, (1, 3))
    reference = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    @test cdf(subset, u) ≈ cdf(reference, u)
    @test pdf(subset, u) ≈ pdf(reference, u)

    # Reordering changes flip positions, not their original dimension labels.
    C13 = SurvivalCopula{3}(ClaytonCopula{3}(2.0), (1, 3))
    reordered = subsetdims(C13, (3, 1))
    reordered_reference = SurvivalCopula{2}(
        ClaytonCopula{2}(2.0), (1, 2))
    @test cdf(reordered, u) ≈ cdf(reordered_reference, u)
    @test pdf(reordered, u) ≈ pdf(reordered_reference, u)
end
