```@meta
CurrentModule = Copulas
```

# [Extreme Value family](@id Extreme_theory)

*Extreme-value copulas* are max-stable dependence models used throughout
multivariate extreme-value theory. In dimension ``d``, an extreme-value copula
``C`` satisfies

```math
C(u_1^t,\ldots,u_d^t)=C(u_1,\ldots,u_d)^t,\qquad t>0.
```

The natural dimension-free representation is the **stable tail dependence
function** (STDF) ``\ell``:

```math
C(\boldsymbol u)
=
\exp\!\left\{-\ell(-\log u_1,\ldots,-\log u_d)\right\}.
```

A valid STDF is convex and one-homogeneous and satisfies

```math
\max_i x_i\le \ell(\boldsymbol x)\le\sum_{i=1}^d x_i.
```

By homogeneity, ``\ell`` can be restricted to the simplex

```math
\Delta_{d-1}
=
\left\{
\boldsymbol w\in[0,1]^d:
\sum_{i=1}^d w_i=1
\right\}
```

to obtain the multivariate Pickands dependence function
``A:\Delta_{d-1}\to\mathbb R``. For
``r=\sum_i x_i`` and ``\boldsymbol w=\boldsymbol x/r``,

```math
A(\boldsymbol w)=\ell(\boldsymbol w),
\qquad
\ell(\boldsymbol x)
=
r\,A\!\left(\frac{\boldsymbol x}{r}\right),
```

with ``\max_i w_i\le A(\boldsymbol w)\le1``. Thus the Pickands
representation itself is not restricted to dimension two.

What is specifically bivariate is the scalar parametrization of the simplex.
For ``d=2``, write ``\boldsymbol w=(t,1-t)``; then
``A:[0,1]\to[1/2,1]`` and

```math
\ell(x_1,x_2)
=
(x_1+x_2)
A\!\left(\frac{x_1}{x_1+x_2}\right),
```

where

```math
\max\{t,1-t\}\le A(t)\le1,\qquad A(0)=A(1)=1.
```

Accordingly, Copulas.jl uses the STDF ``\ell`` as the dimension-independent
EV interface, while `BivariatePickandsTail` names the additional scalar
Pickands capability used by specialized ``d=2`` algorithms.

The bivariate copula therefore has the classical representation

```math
C(u_1,u_2)
=
\exp\!\left\{
\log(u_1u_2)
A\!\left(
\frac{\log u_1}{\log(u_1u_2)}
\right)
\right\}.
```

!!! info "Bivariate and multivariate EV copulas"
    Copulas.jl now uses ``\ell`` as the mathematical EV interface in arbitrary
    dimension while preserving the mature bivariate Pickands machinery.
    Bivariate formulas based on `A`, `dA`, `d²A`, Ghoudi sampling, conditional
    distortions, and family-specific kernels remain available whenever the tail
    provides them.

!!! tip "Think family first, backend second"
    A public constructor identifies the **mathematical family**. It does not ask
    the user to choose a bivariate or multivariate algorithm. Copulas.jl selects
    the appropriate density and sampling representation internally.

## What is new in the EV subsystem?

The current EV implementation goes beyond the historical bivariate-only design:

- the core representation is a dimension-aware STDF `ℓ`;
- `BivariatePickandsTail` identifies a tail with a native scalar bivariate
  Pickands kernel and is a **computational capability**, not necessarily a
  mathematical restriction to dimension two;
- several classical EV families now use the same public constructor in
  ``d=2`` and ``d>2``;
- matrix and vector parameterizations infer the dimension when the parameter
  shape determines it;
- general Hüsler-Reiss variograms and extremal-``t`` correlation matrices are
  supported in addition to exchangeable submodels;
- Tawn and asymmetric Galambos expose structured multivariate subset
  parameterizations;
- BC2 and Marshall-Olkin have multivariate spectral/shock representations;
- multivariate empirical EV estimation is available through a shape-constrained
  discrete spectral projection;
- multivariate densities can be built from mixed partial derivatives of
  ``\ell``;
- `rand(C, n)` uses internal backend routing, so optimized bivariate and
  multivariate samplers coexist behind one public API.

## Constructors and dimensional conventions

The canonical constructor always makes the dimension part of the type:

```julia
FamilyCopula{d}(params...)
```

The equivalent runtime-dimension form

```julia
FamilyCopula(d, params...)
```

is convenient but is not type-stable with respect to `d`.

!!! warning "No implicit bivariate dimension for scalar parameters"
    Scalar/exchangeable EV families require an explicit dimension. For example,
    write `GalambosCopula{2}(2.3)` or `GalambosCopula(2, 2.3)`;
    `GalambosCopula(2.3)` is intentionally not a constructor.

Scalar or exchangeable families therefore have canonical forms

```julia
LogCopula{d}(θ)
GalambosCopula{d}(θ)
MixedCopula{d}(θ)
CuadrasAugeCopula{d}(θ)
HuslerReissCopula{d}(θ)
tEVCopula{d}(ν, ρ)
```

with `FamilyCopula(d, ...)` as runtime sugar.

Structured parameterizations follow exactly the same rule:

```julia
HuslerReissCopula{d}(Γ)
tEVCopula{d}(ν, R)
TawnCopula{d}(α, weights)
AsymGalambosCopula{d}(α, weights)
BC2Copula{d}(a)
MOCopula{d}(λ)
EmpiricalEVCopula{d}(U)
```

Because these objects determine their own dimension, the following additional
convenience forms are also available:

```julia
HuslerReissCopula(Γ)
tEVCopula(ν, R)
TawnCopula(α, weights)
AsymGalambosCopula(α, weights)
BC2Copula(a)
MOCopula(λ)
EmpiricalEVCopula(U)
```

The full subset representations are likewise available as

```julia
TawnCopula{d}(dep, asy)
AsymGalambosCopula{d}(dep, asy)
```

together with `FamilyCopula(d, dep, asy)`. Since `asy` contains exactly
`2^d-1` subset-weight vectors, `TawnCopula(dep, asy)` and
`AsymGalambosCopula(dep, asy)` can infer `d` and construct the same model.

| Family | Canonical constructor | Supported dimension | Interpretation |
|---|---|---:|---|
| Logistic | `LogCopula{d}(θ)` | ``d\ge2`` | exchangeable |
| Galambos | `GalambosCopula{d}(θ)` | ``d\ge2`` | exchangeable negative logistic |
| Mixed | `MixedCopula{d}(θ)` | ``d\ge2`` | scalar Copulas.jl extension of the bivariate mixed model |
| Cuadras-Augé | `CuadrasAugeCopula{d}(θ)` | ``d\ge2`` | scalar |
| Hüsler-Reiss | `HuslerReissCopula{d}(θ)` | ``d\ge2`` | exchangeable variogram |
| Hüsler-Reiss | `HuslerReissCopula{d}(Γ)` | ``d=\mathrm{size}(Γ,1)`` | general variogram |
| extremal-``t`` | `tEVCopula{d}(ν, ρ)` | ``d\ge2`` | equicorrelation |
| extremal-``t`` | `tEVCopula{d}(ν, R)` | ``d=\mathrm{size}(R,1)`` | general correlation matrix |
| Tawn | `TawnCopula{d}(α, weights)` | ``d=\mathrm{length}(weights)`` | full-set logistic component + singleton remainders |
| Tawn | `TawnCopula{d}(dep, asy)` | ``d\ge2`` | full subset representation |
| Asymmetric Galambos | `AsymGalambosCopula{2}(α, θ₁, θ₂)` | 2 | scalar Pickands fast path of the unified subset model |
| Asymmetric Galambos | `AsymGalambosCopula{d}(α, weights)` | ``d=\mathrm{length}(weights)`` | full-set negative-logistic component + singleton remainders |
| Asymmetric Galambos | `AsymGalambosCopula{d}(dep, asy)` | ``d\ge2`` | full subset representation |
| BC2 | `BC2Copula{2}(a, b)` | 2 | classical bivariate representation |
| BC2 | `BC2Copula{d}(a::AbstractVector)` | ``d=\mathrm{length}(a)`` | two-atom spectral representation |
| Marshall-Olkin | `MOCopula{2}(λ₁, λ₂, λ₁₂)` | 2 | classical three-shock representation |
| Marshall-Olkin | `MOCopula{d}(λ)` | ``\mathrm{length}(λ)=2^d-1`` | full subset-shock representation |
| Empirical EV | `EmpiricalEVCopula{2}(U)` | 2 | bivariate Pickands/CFG/OLS estimator |
| Empirical EV | `EmpiricalEVCopula{d}(U)` | ``d=\mathrm{size}(U,1)`` | shape-constrained multivariate spectral estimator |
| Asymmetric logistic | `AsymLogCopula{2}(...)` | 2 | bivariate |
| Asymmetric mixed | `AsymMixedCopula{2}(...)` | 2 | bivariate |

### Exchangeable versus general Hüsler-Reiss and extremal-t

Hüsler-Reiss has two public parameterizations. In the package convention,

```math
\gamma=\left(\frac{2}{\theta}\right)^2
```

maps the exchangeable scalar parameter to the common off-diagonal variogram
entry. Thus

```julia
HuslerReissCopula{d}(θ)
```

is an exchangeable submodel, while

```julia
HuslerReissCopula(Γ)
```

accepts a general valid variogram matrix. In dimension two a matrix
parameterization is validated and then reduced internally to the scalar
bivariate representation. The family seen by the user does not change.

The extremal-``t`` family follows the same pattern:

```julia
tEVCopula{d}(ν, ρ) # equicorrelation
tEVCopula(ν, R)    # general correlation matrix
```

A valid ``2\times2`` correlation matrix is likewise reduced to the specialized
bivariate tail.

!!! info "Public family ≠ internal representation"
    Two constructors of the same mathematical family may store different tail
    types internally. This is deliberate: it lets Copulas.jl retain fast
    bivariate kernels without duplicating the public family.

### Tawn and asymmetric Galambos subset models

The multivariate Tawn model follows the asymmetric logistic construction of
Tawn [tawn1990multivariate](@cite). The full representation associates
components with every nonempty subset of ``\{1,\ldots,d\}``. There are

```math
2^d-1
```

nonempty subsets and

```math
2^d-d-1
```

non-singleton subsets.

Accordingly,

```julia
TawnCopula{d}(dep, asy)
```

contains one dependence parameter per non-singleton subset and one asymmetry
weight vector per nonempty subset. The weights involving each margin must sum
to one.

`TawnCopula(α, weights)` is a convenient lower-dimensional parameterization
implemented in Copulas.jl: one logistic component acts on the full set and
singleton components carry the remaining marginal mass.

Asymmetric Galambos uses the analogous negative-logistic subset construction;
the multivariate min-stable framework is described by Joe [Joe1990](@cite).
The convenience constructor

```julia
AsymGalambosCopula(α, weights)
```

is a Copulas.jl parameterization of that valid subset model, not a separate
literature family.

### Implementation-derived Mixed extension

The historical Mixed model is bivariate [tawn1988bivariate](@cite). The
``d``-dimensional extension used by Copulas.jl is obtained from the identity

```math
\ell_{\mathrm{Mixed},\theta}(\boldsymbol x)
=
(1-\theta)\sum_{i=1}^d x_i
+
\theta\,\ell_{\mathrm{Galambos},1}(\boldsymbol x).
```

Both terms are valid STDFs, so their convex combination is a valid STDF. For
``d=2`` the identity reduces exactly to

```math
A(t)=1-\theta t(1-t),
```

which is the historical Mixed Pickands model.

!!! note "What is literature and what is derived here?"
    Tawn [tawn1988bivariate](@cite) is the reference for the original bivariate
    Mixed model, and Galambos [galambos1975order](@cite) for the
    negative-logistic component. The dimension-free convex-combination identity
    above is the extension used and derived in the Copulas.jl implementation;
    we do not attribute that exact ``d``-dimensional parameterization to either
    paper.

### Multivariate empirical EV estimation

`EmpiricalEVCopula` selects the estimator from the sample dimension. In two
dimensions it preserves the historical Pickands/CFG/OLS estimator of ``A``.

For ``d\ge3``,

```julia
EmpiricalEVCopula(U; method=:ols)
```

first constructs a multivariate Pickands pilot estimator and then projects it
onto the class induced by a finite discrete spectral measure. This matters
because convexity and the elementary Pickands bounds are no longer sufficient
to characterize validity when ``d\ge3``. The multivariate estimators follow
Gudendorf and Segers [gudendorf2011nonparametric](@cite), and the
shape-constrained discrete spectral projection follows
Gudendorf and Segers [gudendorf2012multivariate](@cite).

The resulting object stores a `DiscreteSpectralTail`; consequently the fitted
STDF is valid by construction and exact spectral sampling is available.

!!! info "One public constructor, two internal representations"
    `EmpiricalEVCopula` uses the lightweight historical implementation in two
    dimensions and the shape-constrained spectral representation in higher
    dimensions.

## Advanced Concepts

Here, we present some important concepts from the theory of extreme value copulas that are useful for the development of this package.

Let $(X,Y) \sim C$ where $C$ is a bivariate extreme value copula. We have the following result from [ghoudi1998proprietes](@cite):

!!! property "Ghoudi 1998"
    Let $(X, Y) \sim C$, where $C$ is an extreme value copula. The joint distribution of $X$ and $Z = \frac{\log(X)}{\log(XY)}$ is given by:

    $$P(Z \leq z, X \leq x) = G(z, x) = \left(z + z(1 - z)\frac{A'(z)}{A(z)}\right)x^{A(z)/z}, \quad 0 \leq x, z \leq 1$$

    where $A'(z)$ denotes the derivative of $A(z)$ at point $z.$

Since $A$ is a convex function defined on $[0, 1]$ and satisfies $-1 \leq A'(z) \leq 1$, by extension, we define $A'(1)$ as the supremum of $A'(z)$ over $(0, 1)$. By setting $x = 1$ in the previous result, we obtain the marginal distribution of $Z$:
$$P(Z \leq z) = G_Z(z) = z + z(1 - z) \frac{A'(z)}{A(z)}, \quad 0 \leq z \leq 1.$$

This result was demonstrated by Deheuvels (1991) [deheuvels1991limiting](@cite) in the case where $A$ admits a second derivative.


## Simulation of Bivariate Extreme Value Distributions

To simulate a bivariate extreme value distribution $C(x, y)$, note that if $F_1$ and $F_2$ are univariate extreme value distributions, then the pair $(F_1^{-1}(X), F_2^{-1}(Y))$ is distributed according to a bivariate extreme value distribution. The proposed algorithm in Ghoudi, 1998 [ghoudi1998proprietes](@cite) allows simulating such a distribution.

Assume $A$ has a second derivative, making the distribution absolutely continuous. In this case, $Z$ is also absolutely continuous and has a density $g_Z(z)$ given by:

$$g_Z(z)=1+(1-2z)\frac{A'(z)}{A(z)}
+z(1-z)\left[\frac{A''(z)}{A(z)}
-\left(\frac{A'(z)}{A(z)}\right)^2\right].$$

The conditional distribution of $W$ given $Z$ is:

$$F(w|z) = \frac{1}{g_Z(z)} \frac{d}{dz} F(z, w),$$ 

which simplifies to:

$$F(w|z) = w \frac{z(1 - z) A'(z)}{A(z) g_Z(z)} + (w - w \log w) \left(1 - \frac{z(1 - z) A''(z)}{A(z) g_Z(z)} \right)$$

Given $Z$, the distribution of $W$ is uniform on $(0, 1)$ with probability $p(Z)$ and equals the product of two independent uniforms on $(0, 1)$ with probability $1 - p(Z)$, where:

$$p(z) = \frac{z(1-z)A''(z)}{A(z)g_Z(z)}.$$

Since $g_Z(z)$ is the derivative of the cumulative distribution function of $Z$, it holds that $0 \leq p(z) \leq 1$.

For the class of Extreme Value Copulas, We follow the methodology proposed by Ghoudi,1998. page 191. [ghoudi1998proprietes](@cite). Here, is a detailed algorithm for sampling from bivariate Extreme Value Copulas:

!!! algorithm "Bivariate Extreme Value Copulas sampling"

    * Simulate $U_1, U_2 \sim \mathcal{U}[0, 1]$
    * Simulate $Z \sim G_Z(z)$
    * Select $W = U_1$ with probability $p(Z)$ and $W = U_1U_2$ with probability $1 - p(Z)$
    * Return $X = W^{Z/A(Z)}$ and $Y = W^{(1 - Z)/A(Z)}$  

Note that all functions present in the algorithm were previously defined to ensure that the implemented methodology has a solid theoretical basis.


### Multivariate sampling and backend routing

The bivariate Ghoudi construction above remains an important part of the EV
implementation. It is **not** replaced by multivariate sampling.

The public interface is always

```julia
rand(C, n)
```

and Copulas.jl chooses the backend internally. A tail with a native bivariate
Pickands kernel can use the Ghoudi route in ``d=2``; a family with a faster or
more natural exact multivariate representation can transparently use that
representation even in dimension two.

!!! tip "You never select the sampler yourself"
    `rand(LogCopula(2, θ), n)` and `rand(LogCopula(10, θ), n)` have the same
    public API. The same is true for Galambos, Hüsler-Reiss, Mixed, and
    extremal-``t``. Backend routing is an implementation detail.

This separation is useful because the best algorithm is family-specific:
specialized bivariate Pickands sampling is excellent for some tails, whereas
spectral or max-stable constructions can be dramatically faster for others.

```@docs; canonical=false
Tail
ExtremeValueCopula
```

## Conditionals and distortions

For any copula $C$, the conditional copula and the univariate conditional distortions are given by partial-derivative ratios. If we condition on a set $J$ with $m=|J|$ components and write $I=\{1,\dots,d\}\setminus J$, then

$$C_{I\mid J}(\boldsymbol u_I\mid \boldsymbol u_J)\;=\;\frac{\partial^{m}}{\partial \boldsymbol u_J}\,C(\boldsymbol u_I,\boldsymbol u_J)\,\bigg/\,\frac{\partial^{m}}{\partial \boldsymbol u_J}\,C(\boldsymbol 1_I,\boldsymbol u_J),$$

and, for a single coordinate $i\in I$ (setting the other coordinates in $I$ to 1), the conditional distortion is

$$H_{i\mid J}(u\mid \boldsymbol u_J)\;=\;\frac{\partial^{m}}{\partial \boldsymbol u_J}\,C(\ldots,u_i{=}u,\ldots,\boldsymbol u_J)\,\bigg/\,\frac{\partial^{m}}{\partial \boldsymbol u_J}\,C(\boldsymbol 1,\boldsymbol u_J).$$

For a bivariate extreme value copula with Pickands function $A$, the copula is

$$C(u_1,u_2)=\exp\!\left\{\log(u_1 u_2)\,A\!\left(\frac{\log u_1}{\log(u_1 u_2)}\right)\right\},$$

so the above derivatives can be written explicitly in terms of $A$ (and $A'$ when it exists) by the chain rule. In the implementation, these derivatives are obtained directly from this representation, using analytic formulas when available and automatic differentiation otherwise.

## Visual illustrations

### Pickands dependence functions A(t)

```@example 1
using Copulas, Plots, Distributions
ts = range(0.0, 1.0; length=401)
Cs = (
    GalambosCopula(2, 0.8),    # upper tail dep.
    HuslerReissCopula(2, 1.0), # intermediate
    LogCopula(2, 1.6),         # asymmetric
)
labels = ("Galambos(0.8)", "Hüsler–Reiss(1.0)", "Log(1.6)")
plot(size=(700, 300))
for (i, C) in enumerate(Cs)
    plot!(ts, Copulas.A.(C.tail, ts); label=labels[i])
end
plot!(ts, max.(ts, 1 .- ts); label="bounds", ls=:dash, color=:black)
plot!(ts, ones(length(ts)); label="1", ls=:dot, color=:gray)
```

### Sample scatter (uniform scale)

```@example 1
C = GalambosCopula(2, 1.0)
plot(C, title="Galambos copula sample")
```

### Conditional distortion (EV example)

```@example 1
C = HuslerReissCopula(2, 1.2)
u2 = 0.4
D = condition(C, 2, u2)
ts = range(0.0, 1.0; length=401)
plot(ts, cdf.(Ref(D), ts); xlabel="u", ylabel="H_{1|2}(u|u₂=0.4)",
     title="Conditional distortion for Hüsler–Reiss")
```

### Rosenblatt sanity check (EV)

```@example 1
using StatsBase
U = rand(C, 2000)
S = reduce(hcat, (rosenblatt(C, U[:, i]) for i in 1:size(U,2)))
ts = range(0.0, 1.0; length=401)
EC = [ecdf(S[k, :]) for k in 1:2]
plot(ts, ts; label="Uniform", color=:blue, alpha=0.6, size=(650,300))
plot!(ts, EC[1].(ts); seriestype=:steppost, label="s₁", color=:black)
plot!(ts, EC[2].(ts); seriestype=:steppost, label="s₂", color=:gray)
```

## [Available models](@id available_extreme_models)

### `MTail`
```@docs; canonical=false
MTail
```


### `NoTail`
```@docs; canonical=false
NoTail
```

### `TawnTail`
```@docs; canonical=false
TawnTail
```

### `AsymGalambosTail`
```@docs; canonical=false
AsymGalambosTail
```

### `AsymLogTail`
```@docs; canonical=false
AsymLogTail
```

### `AsymMixedTail`
```@docs; canonical=false
AsymMixedTail
```

### `BC2Tail`
```@docs; canonical=false
BC2Tail
```

### `CuadrasAugeTail`
```@docs; canonical=false
CuadrasAugeTail
```

### `GalambosTail`
```@docs; canonical=false
GalambosTail
```

### `HuslerReissTail`
```@docs; canonical=false
HuslerReissTail
```

### `LogTail`
```@docs; canonical=false
LogTail
```

### `MixedTail`
```@docs; canonical=false
MixedTail
```

### `MOTail`
```@docs; canonical=false
MOTail
```

### `tEVTail`
```@docs; canonical=false
tEVTail
```

### `EmpiricalEVTail`
```@docs; canonical=false
EmpiricalEVTail
```

### `EmpiricalEVMultivariateTail`
```@docs; canonical=false
EmpiricalEVMultivariateTail
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
