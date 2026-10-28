# Correctness obligation: _nearest_correlation computes the nearest valid correlation
# matrix in the Frobenius norm, and the resulting matrix is guaranteed to be strictly
# positive definite for algorithmic stability (cholesky decomposition)

# Conditions for a valid correlation matrix:
# 1. symmetric
# 2. diagonals elements equal 1
# 3. off-diagonal elements are strictly in [-1, 1]
# 4. strictly positive definite

@testset "Nearest valid correlation matrix" begin
    # A matrix that is already a valid correlation matrix must remain unchanged
    A = [1.0 0.7 0.5; 0.7 1.0 0.1; 0.5 0.1 1.0]
    X = Copulas._nearest_correlation(A)
    @test X == A

    # this is the classic indefinite correlation matrix from Higham's papers
    A = Float64[1 1 0; 1 1 1; 0 1 1]
    Y = [1.0 0.76069 0.157298; 0.76069 1.0 0.76069; 0.157298 0.76069 1.0]
    X = Copulas._nearest_correlation(A)
    @test all(p -> isapprox(p[1], p[2]; atol=0.005), zip(X, Y)) # wiggle room for algorithmic variance
    @test all(==(1.0), diag(X))

    # Resulting matrix should have the same eltype as float(eltype(M))
    for T in (Float16, Float32, Float64)
        A = T[1 1 0; 1 1 1; 0 1 1]
        X = Copulas._nearest_correlation(A)
        @test eltype(X) === T
    end
end
