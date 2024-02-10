# Function to generate "n" random samples from an archimedean copula of dimension `dim`
function generate_copula_samples(dim)
    copula = ClaytonCopula(dim, 0.8)
    return rand(copula, 10^6)
end

# Efficiency test for generating samples from an archimedean copula
function test_copula_sampling_efficiency(dim)
    result = @benchmark generate_copula_samples($dim)

    println("Execution time for dimension $dim: ", minimum(result).time)
    println("Memory usage for dimension $dim: ", minimum(result).memory)
    println("\n")
end

dimensions_to_test = [2, 5, 10]

for dim in dimensions_to_test
    println("Evaluating efficiency for dimension $dim...\n")
    test_copula_sampling_efficiency(dim)
end