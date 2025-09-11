@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(0.50, 1.60)) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(2.50, 0.40)) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(3.0, 2.1)) end

@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(1.50, 1.60)) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(2.50, 0.40)) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(5.0, 0.5)) end

@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.AsymGalambosTail(0.35, (0.65, 0.3)))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.HuslerReissTail(0.6))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.HuslerReissTail(1.8))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.LogTail(2.0))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.LogTail(1.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.GalambosTail(2.5))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.GalambosTail(0.7))) end
@testitem "Generic" tags=[:Generic, :ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.GalambosTail(2.5))) end