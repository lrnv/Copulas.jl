using Test
# Test function for plotly_copula
function test_plotly_copula_methods()
    # Create data and copula for testing
    X1 = Gamma(2, 3)
    X2 = Pareto()
    C = ClaytonCopula(2, 3.5)
    D = SklarDist(C, (X1, X2))
    samples = rand(D, 1000)
    
    # Methods to try
    methods = ["pdf", "cdf", "scatter", "scatter+hist", "hist2D", "pdf_contours", "cdf_contours", "scatter3d"]
    
    for method in methods
        println("Testing method: $method")
        
        if method == "scatter3d" && length(D) == 2
            # Test 1: Verify that plotly_copula generates an error for scatter3d with bivariate copula
            @test_throws ErrorException plotly_copula(D, samples, method)
        else
            # Test 1: Verify that plotly_copula does not throw errors
            result = plotly_copula(D, samples, method)
            
           # Test 2: Verify that plotly_copula generates a plot
           @test isa(result, PlotlyJS.SyncPlot)
        end
    end
end


@testset "Graphics Function Tests" begin
    test_plotly_copula_methods()
end

function test_multivariate_copula_methods()
    # Methods to prove
    Methods = ["scatter", "scatter+hist", "hist2D", "pdf_contours", "cdf_contours", "pdf", "cdf"]
    
    for method in Methods
        for i in [4, 5,6]
            D_mult = InvGaussianCopula(i, 2.5)
            samples_mult = rand(D_mult, 300)
            println("Testing method for multivariate copula: $method")
            @test_throws ErrorException plotly_copula(D_mult, samples_mult, method)
        end
    end 
    
    # Trivariate prove
    for method in Methods
        for i in [1, 2, 3]
            T = JoeCopula(3, i)
            samples_tri = rand(T, 1000)
            
            try
                if length(T) == 3 && method == "scatter3d"
                    result = plotly_copula(T, samples_tri, method)
                    @test isa(result, PlotlyJS.SyncPlot)
                else
                    println("Testing method for trivariate copula: $method")
                    @test_throws ErrorException plotly_copula(T, samples_tri, method)
                end
            catch ex
                println("Error in method $method for trivariate copula: $ex")
            end
        end
    end
end


@testset "Tests de funciones de gráficos para copulas multivariadas" begin
    test_multivariate_copula_methods()
end

