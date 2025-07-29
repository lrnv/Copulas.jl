# Fitting compound distributions

Through the `SklarDist` interface, it is possible to fit distributions constructed from a copula and marginals:

```@example 5
using Copulas
using Distributions

# Let's sample some datas:
X₁ = LogNormal()
X₂ = Pareto()
X₃ = Gamma()
X₄ = Normal()
C = SurvivalCopula(FrankCopula(4,7),(2,4))
D = SklarDist(C,(X₁,X₂,X₃,X₄))
data = rand(D,1000)
```

The fit function uses a type as its first argument that describes the structure of the model : 
```@example 5
MyCop = SurvivalCopula{4,ClaytonCopula,(2,4)}
MyMargs = Tuple{LogNormal,Pareto,Gamma,Normal}
MyD = SklarDist{MyCop, MyMargs}
fitted_model = fit(MyD,data)
```

Another possibility is to use an empirical copula and only fit the marginals: 
```@example 5
other_fitted_model = fit(SklarDist{EmpiricalCopula,MyMargs},data)
```

This simple interface leverages the `fit` function from `Distributions.jl`. According to their documentation, this function is not supposed to use a particular method but to fit "quick and dirty" some distributions. 

So you have to be careful: the fit method might not be the same for different copulas or different marginals. For example, Archimedean copulas are fitted through inversion of the Kendall tau function, while the Gaussian copula is fitted by maximum likelihood.