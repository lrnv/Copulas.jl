```@meta
CurrentModule = Copulas
```

# Other Copulas

Some copulas, while necessary in certain cases and very useful, are hard to classify. We gather them here for simplicity. 

## [Independence and Fréchet-Hoeffding bounds](@id bestiary_ref)

### `IndependentCopula`

The independence copula is

```math
\Pi_d(\boldsymbol u)=\prod_{j=1}^d u_j.
```

It has no free parameter and is constructed with `IndependentCopula(d)` or
`IndependentCopula{d}()`.

### `MCopula`

The upper Fréchet--Hoeffding bound, or comonotonic copula, is
``M_d(\boldsymbol u)=\min_j u_j``. It is parameter-free and singular, and is
constructed with `MCopula(d)` or `MCopula{d}()`.

### `WCopula`

The lower Fréchet--Hoeffding bound is a copula only in dimension two:
``W(u,v)=\max(u+v-1,0)``. This countermonotonic, singular model is constructed
with `WCopula()`, `WCopula(2)`, or `WCopula{2}()`.

## Transformed Copulas

### `SurvivalCopula`

If ``\boldsymbol U\sim C`` and ``J`` is a set of coordinates, the survival
transformation is the copula of ``\boldsymbol V`` defined by
``V_j=1-U_j`` for ``j\in J`` and ``V_j=U_j`` otherwise. It preserves the
dimension and has no additional continuous parameter. Construct it with
`SurvivalCopula(C, flips)`, where `flips` contains distinct indices in
`1:length(C)`.

When fitting a rotated model, pass the desired flip indices explicitly because
they belong to the instance rather than its type:

```julia
using Distributions
S = SurvivalCopula(ClaytonCopula(2, 2.0), (1,))
U = rand(S, 100)
Ŝ = fit(typeof(S), U; flips=(1,))
```

## Others

### `PlackettCopula`

The bivariate Plackett family is

```math
C_\theta(u,v)=\frac{1+(\theta-1)(u+v)
-\sqrt{[1+(\theta-1)(u+v)]^2-4\theta(\theta-1)uv}}
{2(\theta-1)},
```

with its continuous value ``uv`` at ``\theta=1``. The parameter satisfies
``\theta\ge0``; zero and infinity give the lower and upper
Fréchet--Hoeffding bounds. Use `PlackettCopula(θ)`,
`PlackettCopula{2}(θ)`, or `PlackettCopula(2, θ)`; see [nelsen2006](@cite).

### `FGMCopula`

For every subset ``S\subseteq\{1,\ldots,d\}`` with ``|S|\ge2``, let
``\theta_S`` be an interaction parameter. The multivariate
Farlie--Gumbel--Morgenstern copula is

```math
C(\boldsymbol u)=\prod_{j=1}^d u_j
\left[1+\sum_{|S|\ge2}\theta_S\prod_{j\in S}(1-u_j)\right].
```

The parameter vector therefore has length ``2^d-d-1``. Each coefficient lies
in ``[-1,1]`` and the joint corner constraints ensuring a non-negative density
must also hold. Construct the family with `FGMCopula{d}(θ)` or
`FGMCopula(d, θ)`; in dimension two, `θ` may be supplied as a scalar. The
sampling representation follows [blier2022stochastic](@cite). Even at the
bivariate endpoints `θ = ±1`, these remain ordinary, weak-dependence FGM
copulas rather than either Fréchet--Hoeffding bound.

### `RafteryCopula`

Writing ``u_{(1)}\le\cdots\le u_{(d)}`` for the ordered coordinates, the
Raftery family is

```math
C_\theta(\boldsymbol u)=u_{(1)}
+\frac{(1-\theta)(1-d)}{1-\theta-d}
 \left(\prod_{j=1}^d u_j\right)^{1/(1-\theta)}
-\sum_{i=2}^d
 \frac{\theta(1-\theta)}{(1-\theta-i)(2-\theta-i)}
 \left(\prod_{j=1}^{i-1}u_{(j)}\right)^{1/(1-\theta)}
 u_{(i)}^{(2-\theta-i)/(1-\theta)}.
```

Here ``0\le\theta\le1``; the endpoints give independence and
comonotonicity. Use `RafteryCopula{d}(θ)` or `RafteryCopula(d, θ)`; the
multivariate extension is described in [Raftery2023](@cite).

See the canonical [Public API](@ref) for detailed validation and limiting
behavior of these constructors.

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
