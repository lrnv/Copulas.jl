@testitem "Ejemplo de uso" begin
    using Distributions


    # This one works correctly: 
    G = GalambosCopula(2,2.5)
    u = [0.5, 0.6]
    cdf(G,u) 
    pdf(G,u)
    # Although I am not sure about the produced values ? 

    # This one fails : 
    # Ejemplo de uso
    C = TEVCopula(4, [1.0 0.5; 0.5 1.0])
    u = [0.4, 0.5]
    cdf(C,u)
    pdf(C,u)


end