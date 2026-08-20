# Resolve constructor intersections for the EV boundary tails.
ExtremeValueCopula(d::Int, ::NoTail) = IndependentCopula(d)
ExtremeValueCopula(d::Int, ::MTail) = MCopula(d)
