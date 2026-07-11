```@meta
CurrentModule = Copulas
```

# Vines family

One notable class of copulas is the vine copulas. These distributions use a
graph of conditional distributions to encode the distribution of the random
vector. To define such a model, working with conditional densities, and given
any ordered partition $\boldsymbol i_1, ..., \boldsymbol i_p$ of $1, ..., d$, we
write:

$$f(\boldsymbol x) = f(x_{\boldsymbol i_1}) \prod\limits_{j=1}^{p-1} f(x_{\boldsymbol i_{j+1}} | x_{\boldsymbol i_j}).$$

Of course, the choice of partition, its order, and the conditional models is
left to the practitioner. The goal when dealing with such dependency graphs is
to tailor the graph to reduce the approximation error, which can be a
challenging task. There exist simplifying assumptions that help with this, and
we refer to [durante2017a, nagler2016, nagler2018, czado2013, czado2019,
graler2014](@cite) for a deep dive into vine theory, along with results and
extensions.

Vine copulas are available through the add-on package
[`VineCopulas.jl`](https://github.com/Santymax98/VineCopulas.jl), which is built
on top of `Copulas.jl`. It provides explicit C-vine, D-vine, and regular-vine
models, with density evaluation, simulation, Rosenblatt transforms, pair-copula
conditional primitives, and R-vine matrix helpers. See the
[`VineCopulas.jl` documentation](https://santymax98.github.io/VineCopulas.jl/dev/)
for the full API and current scope.

## A small D-vine example

```@example vines
using Copulas
using VineCopulas
using Distributions: logpdf, pdf
using Random

C12 = GaussianCopula([1.0 0.5; 0.5 1.0])
C23 = ClaytonCopula(2, 2.0)
C13_2 = FrankCopula(2, 3.0)

vine = DVineCopula([1, 2, 3], [[C12, C23], [C13_2]])

u = [0.2, 0.5, 0.7]
(logpdf(vine, u), pdf(vine, u))
```

```@example vines
U = rand(MersenneTwister(123), vine, 5)
size(U)
```

Matrices follow the same convention as in `Copulas.jl`: rows are dimensions and
columns are observations. See the
[`VineCopulas.jl` documentation](https://santymax98.github.io/VineCopulas.jl/dev/)
for the full API and current scope.

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
