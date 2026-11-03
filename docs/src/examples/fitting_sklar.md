# Fitting compound distributions

`SklarDist` models can be fitted sequentially with IFM or jointly with JMLE.
The default remains IFM.

```@example fitting_sklar
using Copulas, Distributions, Random

truth = SklarDist(ClaytonCopula{2}(1.2), (Normal(0.5, 1.0), Exponential(0.7)))
X = rand(Xoshiro(42), truth, 300)
T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}

M_ifm = fit(CopulaModel, T, X; sklar_method=:ifm)
M_jmle = fit(CopulaModel, T, X; sklar_method=:jmle)
```

When an initial `SklarDist` is available, it can be passed directly to JMLE:

```@example fitting_sklar
S0 = SklarDist(ClaytonCopula{2}(0.8), (Normal(), Exponential()))
M0 = fit(CopulaModel, S0, X; sklar_method=:jmle)
fitted_distribution(M0)
```

The bootstrap refits the complete recorded estimator, including JMLE:

```@example fitting_sklar
I = infer(M_jmle; method=:bootstrap, nresamples=10, rng=Xoshiro(43))
```
