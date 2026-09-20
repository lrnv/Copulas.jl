# [Liebscher and Khoudraji copulas](@id liebscher_copulas)

Liebscher copulas combine several copulas through coordinate-wise power transformations while preserving uniform margins. Copulas.jl implements the power Liebscher construction

```math
C(\boldsymbol u) = \prod_{k=1}^{K} C_k\left(u_1^{a_{k1}},\ldots,u_d^{a_{kd}}\right),
```

where `C_1,\ldots,C_K` are `d`-dimensional copulas and the non-negative weights satisfy

```math
a_{kj}\ge 0, \qquad \sum_{k=1}^{K} a_{kj}=1, \qquad j=1,\ldots,d.
```

The public constructor is

```julia
LiebscherCopula(d, copulas, weights)
```

where `copulas` is a tuple containing the component copulas and `weights` is a `K × d` matrix. Each row corresponds to one component copula and each column to one coordinate.

## Constructing a Liebscher copula

A bivariate model can be built from two different copula families:

```@example liebscher
using Copulas, Distributions, Random

C1 = ClaytonCopula(2, 2.0)
C2 = GumbelCopula(2, 1.5)

W = [0.3 0.8; 0.7 0.2]

C = LiebscherCopula(2, (C1, C2), W)

u = [0.4, 0.7]

(cdf = cdf(C, u), logpdf = logpdf(C, u))
```

The column-sum restriction on `W` guarantees that every marginal remains uniform. For example, the first coordinate receives weights `0.3` and `0.7`, while the second receives `0.8` and `0.2`.

The construction extends directly to arbitrary dimension and any number of components:

```@example liebscher
C1_3 = ClaytonCopula(3, 1.5)
C2_3 = GumbelCopula(3, 1.4)

W3 = [0.2 0.5 0.7; 0.8 0.5 0.3]

C3 = LiebscherCopula(3, (C1_3, C2_3), W3)

cdf(C3, [0.3, 0.6, 0.8])
```

Zero weights deactivate the corresponding coordinate of a component. A component whose whole row is zero has no effect on the resulting copula.

## Khoudraji construction

The Khoudraji construction is the two-component power Liebscher model

```math
C_K(\boldsymbol u) = C_1\left(u_1^{1-\alpha_1},\ldots,u_d^{1-\alpha_d}\right)C_2\left(u_1^{\alpha_1},\ldots,u_d^{\alpha_d}\right),
```

with

```math
0\le \alpha_j\le 1, \qquad j=1,\ldots,d.
```

Copulas.jl exposes this parameterization through

```julia
KhoudrajiCopula(d, (C1, C2), shapes)
```

where `shapes[j] = α_j`.

```@example liebscher
K = KhoudrajiCopula(2, (C1, C2), [0.25, 0.8])

K isa LiebscherCopula
```

`KhoudrajiCopula` is a convenience constructor rather than a distinct copula type. Internally it creates a `LiebscherCopula` with weight matrix

```math
\begin{pmatrix}
1-\alpha_1 & \cdots & 1-\alpha_d \\
\alpha_1 & \cdots & \alpha_d
\end{pmatrix}.
```

When only one component copula is supplied,

```julia
KhoudrajiCopula(d, C, shapes)
```

the first component is the independence copula:

```math
C_K(\boldsymbol u) = \Pi\left(u_1^{1-\alpha_1},\ldots,u_d^{1-\alpha_d}\right)C\left(u_1^{\alpha_1},\ldots,u_d^{\alpha_d}\right).
```

```@example liebscher
Kasym = KhoudrajiCopula(2, C1, [0.25, 0.8])

cdf(Kasym, [0.4, 0.7])
```

This form is particularly useful for introducing asymmetry into an otherwise symmetric copula family.

## Sampling

The power Liebscher construction admits an exact sampling representation. For independent vectors

```math
\boldsymbol X^{(k)} \sim C_k,
```

the resulting coordinates can be generated as

```math
U_j=\max_{k:a_{kj}>0}\left(X_j^{(k)}\right)^{1/a_{kj}}.
```

Copulas.jl uses this representation directly for random generation:

```@example liebscher
rng = Xoshiro(42)

U = rand(rng, C3, 100)

size(U)
```

The resulting sample has uniform margins and dependence determined by the component copulas and their coordinate weights.

## Subsetting

Marginalization preserves the Liebscher structure. Selecting a set of coordinates keeps the corresponding columns of the weight matrix and subsets every component copula to the same coordinates.

```@example liebscher
C31 = subsetdims(C3, (3, 1))

last(params(C31))
```

The order of the requested coordinates is preserved:

```@example liebscher
u31 = [0.7, 0.4]

cdf(C31, u31)
```

A one-dimensional subset returns the ordinary uniform distribution, as for every copula in Copulas.jl.

## Conditioning and Rosenblatt transforms

Liebscher copulas use the generic conditioning machinery through exact mixed partial derivatives of the product construction. The usual public interface is therefore available without introducing a separate conditional family.

```@example liebscher
conditional_23_given_1 = condition(C3, 1, 0.4)

cdf(conditional_23_given_1, [0.5, 0.7])
```

Successive conditional distributions also provide Rosenblatt and inverse Rosenblatt transforms:

```@example liebscher
u = [0.3, 0.5, 0.8]

v = rosenblatt(C3, u)

u_again = inverse_rosenblatt(C3, v)

maximum(abs, u_again - u)
```

The same operations are available for models created with `KhoudrajiCopula`, since those objects are ordinary `LiebscherCopula` instances.

## Absolute continuity

If every active multivariate component is absolutely continuous, the resulting Liebscher copula has an ordinary Lebesgue density and supports `pdf` and `logpdf`.

```@example liebscher
pdf(C, [0.4, 0.7])
```

Components that are singular only through coordinates with zero Liebscher weight do not affect the measure class. If an active component introduces a singular multivariate contribution, the resulting model is treated as non-absolutely continuous and no global Lebesgue density is exposed.

## Limiting cases

Several useful cases follow directly from the weight matrix:

* A zero weight removes one coordinate from the corresponding component factor.
* A row of zeros makes that entire component inactive.
* With one component and weights equal to one, the Liebscher construction reproduces that component copula.
* If all component copulas are independent, the resulting copula is independent.
* `KhoudrajiCopula(d, C, zeros(d))` reduces to the independence copula.
* `KhoudrajiCopula(d, C, ones(d))` reduces to `C`.

These identities also make the Khoudraji construction convenient for continuously interpolating between independence and a selected dependence model while allowing different coordinates to receive different amounts of asymmetry.
