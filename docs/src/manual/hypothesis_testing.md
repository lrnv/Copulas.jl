```@meta
CurrentModule = Copulas
```

# [Hypothesis testing](@id hypothesis_testing)

`Copulas.jl` provides a common interface for rank-based hypothesis tests on copulas. The framework separates three distinct ingredients:

1. the **null hypothesis** being tested;
2. the **test statistic** used to measure departures from the null;
3. the **calibration method** used to obtain a p-value.

Conceptually,

```text
CopulaHypothesis
      ×
  Statistic
      ×
 Calibration
      ↓
  CopulaTest
```

This separation makes it possible to reuse the same calibration machinery across different hypotheses and to introduce new statistics or hypotheses without modifying the generic test constructor.

The current implementation includes tests of:

* mutual independence;
* exchangeability;
* radial symmetry;
* extreme-value dependence (max-stability);
* goodness of fit for a specified copula;
* goodness of fit for a fitted copula family.

The procedures are based on empirical-copula processes, resampling, multiplier methods, and parametric bootstrap ideas developed throughout the copula-testing literature; see, among others,

[genest2004independence](@cite),
[fermanian2004empirical](@cite),
[remillard2009equality](@cite),
[bucher2010bootstrap](@cite), and
[genest2009gof](@cite).

---

## Data convention

As elsewhere in `Copulas.jl`, observations are represented by a `d\times n` matrix

```math
U =
\begin{pmatrix}
U_{11} & \cdots & U_{1n}\\
\vdots &        & \vdots\\
U_{d1} & \cdots & U_{dn}
\end{pmatrix},
```

where each column

```math
\boldsymbol U_i = (U_{1i},\ldots,U_{di})^\top
```

is one `d`-dimensional observation.

Hypothesis tests are rank based. When

```julia
pseudo_values=false
```

the input matrix is transformed internally with [`pseudos`](@ref). If the data already consist of pseudo-observations in `[0,1]^d`, use

```julia
pseudo_values=true
```

to avoid ranking them again.

::: warning Continuous margins and ties

The currently implemented copula hypothesis tests assume continuous margins and therefore require tie-free observations in every margin. Tied or discrete data are rejected with an `ArgumentError`.

This is intentional: ordinal ranking would otherwise assign distinct ranks to tied observations and could produce apparently valid p-values without the tie-aware empirical-process or bootstrap theory required for such data. Tie-aware procedures are outside the scope of the current implementation.

:::

Given pseudo-observations $\boldsymbol U_1,\ldots,\boldsymbol U_n$, the empirical copula is

```math
C_n(\boldsymbol u)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1
\left(
\boldsymbol U_i\le\boldsymbol u
\right),
```

where the inequality is understood componentwise. 

Empirical-copula processes and their weak convergence form the theoretical basis for many of the statistics and multiplier approximations used below [fermanian2004empirical](@cite).

---

## Common interface

All tests return a [`CopulaTest`](@ref), which implements `StatsAPI.HypothesisTest`.

For example,

```@example hypothesis_testing
using Copulas, Distributions, Random, StatsBase

U = rand(Xoshiro(123), ClaytonCopula(2, 3.0), 80)

test = IndependenceCopulaTest(U; N=49, rng=Xoshiro(123),)

nothing # hide
```

The common result interface is

```@example hypothesis_testing
teststatistic(test)
```

```@example hypothesis_testing
pvalue(test)
```

```@example hypothesis_testing
nobs(test)
```

A test also records:

* `test.statistic`: the statistic used;
* `test.calibration`: the calibration method;
* `test.n_resamples`: number of resampling replicates;
* `test.dimension`: dimension of the copula;
* `test.details`: test-specific metadata.

Printing the object gives a summary of the hypothesis, statistic, calibration, p-value, and relevant test-specific information.

```@example hypothesis_testing
test
```

!!! note "Number of resamples"
The small values of `N` used in the documentation keep the examples fast.
For statistical work, substantially larger values should generally be used,
depending on the desired Monte Carlo precision.

---

# Mutual independence

## Null hypothesis

Let `C` denote the copula of the random vector. Mutual independence is equivalent to the product copula

```math
\Pi(\boldsymbol u)
=
\prod_{j=1}^{d}u_j.
```

Thus

```math
H_0:
C(\boldsymbol u)
=
\Pi(\boldsymbol u)
\qquad
\text{for every }
\boldsymbol u\in[0,1]^d.
```

Rank-based independence tests constructed from the empirical copula process are studied in [genest2004independence](@cite).

## Cramér--von Mises statistic

The statistic currently available in `Copulas.jl` is `:cvm`. The implementation evaluates the squared discrepancy between the empirical copula and the product copula at the observed pseudo-observations:

```math
S_n^{\mathrm{ind}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
\prod_{j=1}^{d}U_{ji}
\right]^2.
```

Large values indicate departure from mutual independence.

## Calibration

Under `H_0`, the coordinates are independent uniforms. The default calibration is therefore `:simulation`:

1. generate `n` observations from the `d`-dimensional product copula;
2. transform the generated sample to pseudo-observations;
3. recompute `S_n^{\mathrm{ind}}`;
4. repeat the procedure `N` times;
5. compare the observed statistic with its simulated null distribution.

## Usage

```@example hypothesis_testing
Uind = rand(Xoshiro(1), IndependentCopula(3), 100)

tind = IndependenceCopulaTest(Uind; N=49, rng=Xoshiro(2),)

(tind.statistic, tind.calibration, pvalue(tind))
```

The current defaults are

```text
statistic   = :cvm
calibration = :simulation
```

---

# Exchangeability

## Null hypothesis

A copula `C` is exchangeable when it is invariant under permutations of its coordinates.

For a permutation

```math
\pi:
\{1,\ldots,d\}
\longrightarrow
\{1,\ldots,d\},
```

write

```math
\boldsymbol u_\pi
=
(u_{\pi(1)},\ldots,u_{\pi(d)}).
```

Full exchangeability means

```math
H_0:
C(\boldsymbol u)
=
C(\boldsymbol u_\pi)
```

for every `\boldsymbol u\in[0,1]^d` and every coordinate permutation `\pi`.

Empirical-copula tests for bivariate symmetry were developed in [genest2012symmetry](@cite) and extended to arbitrary dimension by [harder2017exchangeability](@cite).

## Statistic

For a collection `\mathcal G` of non-identity permutations, the implemented statistic is

```math
S_n^{\mathrm{ex}}
=
\frac{1}{n}
\sum_{\pi\in\mathcal G}
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_n(\boldsymbol U_{i,\pi})
\right]^2
w_\pi(\boldsymbol U_i).
```

The default weight is `weight=:wm2`.

Let

```math
m(\boldsymbol u)
=
\min_{1\le j\le d}u_j,
```

and

```math
b(\boldsymbol u)
=
d-1+m(\boldsymbol u)-\sum_{j=1}^{d}u_j.
```

For a transposition exchanging coordinates `a` and `b`, define

```math
\omega_\pi(\boldsymbol u)
=
|u_a-u_b|.
```

For a general permutation, let

```math
u_{(1)}\le\cdots\le u_{(d)}
```

denote the ordered coordinates and define the implementation's permutation separation term by

```math
\omega_\pi(\boldsymbol u)
=
\sum_{k=\lceil d/2\rceil+1}^{d}
\left(
u_{(k)}-m(\boldsymbol u)
\right).
```

The `:wm2` weight is then

```math
w_\pi(\boldsymbol u)
=
\left[
\max
\left\{
0,
\min
\left(
m(\boldsymbol u),
\omega_\pi(\boldsymbol u),
b(\boldsymbol u)
\right)
\right\}
\right]^2.
```

Alternatively,

```julia
weight=:none
```

sets `w_\pi(\boldsymbol u)=1`.

## Permutation generators

The keyword `permutations` controls the set `\mathcal G`.

### `permutations=:G2`

This is the default.

For `d=2`, it contains the only nontrivial transposition,

```math
(12).
```

For `d>2`, it uses the transposition

```math
(12)
```

together with the cyclic left shift

```math
(12\cdots d).
```

### `permutations=:G1`

Uses the transpositions

```math
(12),(13),\ldots,(1d).
```

### `permutations=:all`

Uses all non-identity permutations.

A custom permutation or collection of permutations can also be supplied directly.

## Multiplier calibration

The default calibration is `:multiplier`.

The empirical-copula process has a nontrivial correction caused by replacing the unknown margins with ranks. The implementation therefore constructs the corresponding multiplier representation, including finite-difference estimates of the partial derivatives of $C_n$.

The derivative bandwidth is

```math
h_n=n^{-1/2}.
```

For coordinate $j$, the derivative is approximated by a boundary-corrected finite difference of the form

```math
\dot C_{n,j}(\boldsymbol u)
\approx
\frac{
C_n(\boldsymbol u+h_n\boldsymbol e_j)
-
C_n(\boldsymbol u-h_n\boldsymbol e_j)
}{
\text{effective width}
}.
```

Independent exponential multipliers are generated and centered before applying the empirical-process representation. This type of multiplier approximation is closely related to the methods discussed in [remillard2009equality](@cite), [bucher2010bootstrap](@cite), and [harder2017exchangeability](@cite).

## Usage

```@example hypothesis_testing
Uex = rand(Xoshiro(4), GumbelCopula(3, 2.0), 80)

tex = ExchangeabilityCopulaTest(Uex; permutations=:G2, weight=:wm2, N=49, rng=Xoshiro(5),)

(tex.statistic, tex.calibration)
```

The current defaults are

```text
statistic   = :Sn
calibration = :multiplier
```

---

# Radial symmetry

## Null hypothesis

A copula is radially symmetric when

```math
\boldsymbol U
\overset{d}{=}
\boldsymbol 1-\boldsymbol U.
```

Equivalently, if $C^{\mathrm{rad}}$ denotes the copula of $\boldsymbol 1-\boldsymbol U$, then

```math
H_0:
C
=
C^{\mathrm{rad}}.
```

Nonparametric tests of copula symmetry and randomization procedures based on the corresponding invariance group are studied in [beare2020symmetry](@cite).

## Statistic

Let $C_n$ denote the empirical copula of the original pseudo-observations and let $\bar{C_n}$ denote the empirical copula constructed from

```math
\boldsymbol 1-\boldsymbol U_1,
\ldots,
\boldsymbol 1-\boldsymbol U_n.
```

The implemented statistic is

```math
S_n^{\mathrm{rad}}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
\bar C_n(\boldsymbol U_i)
\right]^2.
```

Large values indicate radial asymmetry.

## Randomization calibration

Under radial symmetry, an observation and its radial reflection are distributionally equivalent. For every observation $i$, independently generate

```math
B_i\sim\operatorname{Bernoulli}(1/2),
```

and construct

```math
\boldsymbol U_i^\star
=
\begin{cases}
\boldsymbol U_i,
&
B_i=0,\\[2mm]
\boldsymbol 1-\boldsymbol U_i,
&
B_i=1.
\end{cases}
```

The randomized sample is converted back to pseudo-observations before the statistic is evaluated. Thus the default reflection probability is exactly

```math
\Pr(B_i=1)=\frac12.
```

The procedure exploits the group invariance associated with radial symmetry, following the randomization-testing principle developed in [beare2020symmetry](@cite).

## Usage

```@example hypothesis_testing
Urad = rand(Xoshiro(6), GaussianCopula([1.0 0.6; 0.6 1.0]), 80)

trad = RadialSymmetryCopulaTest(Urad; N=49, rng=Xoshiro(7),)

(trad.statistic, trad.calibration, trad.details.reflection_probability)
```

The current defaults are

```text
statistic   = :Sn
calibration = :randomization
```

---

# Extreme-value dependence

## Max-stability

Extreme-value copulas are characterized by max-stability. For any $r>0$,

```math
C(u_1^r,\ldots,u_d^r)
=
C(u_1,\ldots,u_d)^r.
```

Equivalently, for $r>1$,

```math
C(\boldsymbol u)
=
C(\boldsymbol u^{1/r})^r,
```

where

```math
\boldsymbol u^{1/r}
=
(u_1^{1/r},\ldots,u_d^{1/r}).
```

This characterization provides a direct way to test

```math
H_0:
C\text{ belongs to the extreme-value class}.
```

Large-sample tests based on this max-stability identity, the empirical copula, and multiplier approximations are developed by [kojadinovic2011extremevalue](@cite).

## Statistic

For a finite collection of powers

```math
\mathcal R
=
\{r_1,\ldots,r_K\},
\qquad
r_k>1,
```

the implemented statistic is

```math
S_n^{\mathrm{EV}}
=
\sum_{r\in\mathcal R}
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i^{1/r})^r
-
C_n(\boldsymbol U_i)
\right]^2.
```

The default powers are

```math
\mathcal R=\{3,4,5\}.
```

They can be changed through the `powers` keyword.

## Multiplier calibration

Approximate p-values are obtained from a multiplier representation of the empirical-copula process, following the max-stability testing strategy in
[kojadinovic2011extremevalue](@cite).

As in the exchangeability test, the finite-difference bandwidth used for the empirical partial derivatives is

```math
h_n=n^{-1/2}.
```

The multiplier variables are exponential and centered before the bootstrap process is evaluated.

## Usage

```@example hypothesis_testing
Uev = rand(Xoshiro(8), GumbelCopula(2, 2.5), 80)

tev = ExtremeValueCopulaTest(Uev; powers=3:5, N=49, rng=Xoshiro(9),)

(tev.statistic, tev.calibration, tev.details.powers)
```

A single power is also allowed:

```julia
ExtremeValueCopulaTest(U; powers=2)
```

All supplied powers must be finite and strictly larger than one.

The current defaults are

```text
statistic   = :Sn
calibration = :multiplier
```

---

# Goodness of fit

Copula goodness-of-fit procedures compare the empirical dependence structure with a proposed parametric copula model. Empirical-process and Cramér--von Mises procedures of this form are reviewed extensively in [genest2009gof](@cite).

`Copulas.jl` distinguishes a **simple** null hypothesis from a **composite** null hypothesis.

---

## Simple goodness of fit

Suppose that a fully specified copula `C_0` is given, including all its parameters.

The null hypothesis is

```math
H_0:
C=C_0.
```

The implemented Cramér--von Mises-type statistic is

```math
S_n^{\mathrm{GOF}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_0(\boldsymbol U_i)
\right]^2.
```

Use

```@example hypothesis_testing
C0 = ClaytonCopula(2, 3.0)
Ugof = rand(Xoshiro(10), C0, 80)

tsimple = GOFCopulaTest(C0, Ugof; N=49, rng=Xoshiro(11),)

tsimple.hypothesis.kind
```

which produces a `:simple` goodness-of-fit hypothesis.

### Parametric bootstrap

For every bootstrap replicate:

1. generate `n` observations from `C_0`;
2. transform the sample to pseudo-observations;
3. compute the same goodness-of-fit statistic;
4. compare the bootstrap statistic with the observed value.

No parameters are re-estimated because `C_0` is fully specified.

---

## Composite goodness of fit

Suppose instead that

```math
\mathcal C
=
\{
C_\theta:\theta\in\Theta
\}
```

is a parametric copula family and that $\widehat\theta$ is estimated from the data.

The null hypothesis becomes

```math
H_0:
C\in\mathcal C,
```

and the observed statistic is

```math
S_n^{\mathrm{GOF}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_{\widehat\theta}(\boldsymbol U_i)
\right]^2.
```

Because $\widehat\theta$ is estimated, the uncertainty introduced by fitting must also be reproduced in the bootstrap. Parametric-bootstrap validity for this type of semiparametric goodness-of-fit problem is studied in [genest2008bootstrap](@cite); practical copula GOF procedures and their finite sample behavior are discussed in [genest2009gof](@cite).

In `Copulas.jl`, every composite bootstrap replicate performs the following steps:

```math
\boldsymbol U_1^\star,\ldots,\boldsymbol U_n^\star
\sim
C_{\widehat\theta},
```

then refits the **same copula family**,

```math
\widehat\theta^\star
=
\operatorname{fit}
\left(
\boldsymbol U_1^\star,\ldots,\boldsymbol U_n^\star
\right),
```

and computes

```math
S_n^\star
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
C_n^\star(\boldsymbol U_i^\star)
-
C_{\widehat\theta^\star}(\boldsymbol U_i^\star)
\right]^2.
```

Thus parameter estimation is repeated inside every bootstrap replicate rather than treating the fitted parameters as fixed.

## Usage

First fit a model:

```@example hypothesis_testing
M = fit(CopulaModel, ClaytonCopula, Ugof; vcov=false,)

nothing # hide
```

Then run the test directly from the fitted model:

```@example hypothesis_testing
tcomposite = GOFCopulaTest(M; N=49, rng=Xoshiro(12),)

(tcomposite.hypothesis.kind, pvalue(tcomposite))
```

`GOFCopulaTest(M)` uses the pseudo-observations stored in `M.method_details`.

The equivalent explicit-data form is

```julia
GOFCopulaTest(M, U)
```

when a different data matrix is to be tested against the fitted family.

The current defaults are

```text
statistic   = :Sn
calibration = :parametric_bootstrap
```

---

# Statistics and calibrations

The public keywords

```julia
statistic=:default
calibration=:default
```

are resolved through capability declarations.

For a hypothesis `h`, the available statistics are declared by

```julia
Copulas._available_statistics(h)
```

and the available calibrations for a statistic `s` by

```julia
Copulas._available_calibrations(h, Val(s))
```

The **first element** of each returned tuple is the default.

For example, the independence hypothesis declares conceptually

```julia
_available_statistics(::IndependenceHypothesis) = (:cvm,)

_available_calibrations(::IndependenceHypothesis, ::Val{:cvm},) = (:simulation,)
```

while the extreme-value hypothesis declares

```julia
_available_statistics(::ExtremeValueHypothesis) = (:Sn,)

_available_calibrations(::ExtremeValueHypothesis, ::Val{:Sn},) = (:multiplier,)
```

This convention deliberately mirrors the fitting interface:

```text
_available_fitting_methods
          ↓
    first = default
          ↓
_fit(..., Val(method))
```

and, for hypothesis tests,

```text
_available_statistics
          ↓
    first = default
          ↓
_teststatistic(..., Val(statistic))
```

followed by

```text
_available_calibrations
          ↓
    first = default
          ↓
_calibrate(..., Val(calibration), Val(statistic))
```

The generic `CopulaTest` constructor therefore does not need to know which statistics are implemented by any particular hypothesis.

!!! info "Why use `Val` internally?"
Users interact with ordinary symbols such as `:Sn`, `:cvm`, `:simulation`,
and `:multiplier`. Internally those symbols are converted to `Val` objects,
allowing Julia's multiple dispatch to select the appropriate mathematical
implementation without central `if`/`elseif` tables.

---

# Calibration engines

The framework currently provides four reusable calibration mechanisms.

| Calibration             | Principle                                         | Typical use                               |
| ----------------------- | ------------------------------------------------- | ----------------------------------------- |
| `:simulation`           | Generate directly under `H_0`                     | Independence                              |
| `:randomization`        | Exploit invariance under `H_0`                    | Radial symmetry                           |
| `:multiplier`           | Approximate an empirical-copula process           | Exchangeability, extreme-value dependence |
| `:parametric_bootstrap` | Simulate from a parametric fitted/specifed copula | Goodness of fit                           |

The empirical-copula multiplier methodology is related to [remillard2009equality](@cite) and [bucher2010bootstrap](@cite), while the parametric-bootstrap framework for composite goodness-of-fit hypotheses is studied in [genest2008bootstrap](@cite).

The concrete hypothesis only provides the mathematical ingredients required by the selected engine. The mechanics of repeated simulation, randomization, multiplier generation, or parametric bootstrap remain centralized.

---

# Monte Carlo p-values

Let $T_n$ be the observed statistic and let

```math
T_n^{(1)},\ldots,T_n^{(N)}
```

denote resampled statistics.

For calibrations using the finite-sample correction in the generic engine, `Copulas.jl` computes

```math
\widehat p
=
\frac{
1/2+
\sum_{b=1}^{N}
\mathbf 1
\left\{
T_n^{(b)}\ge T_n
\right\}
}{
N+1
}.
```

Specific calibration methods may override the comparison convention when their theoretical construction requires it. In particular, the exchangeability multiplier implementation uses strict exceedances and its corresponding uncorrected empirical proportion.

Accordingly, $N$ controls Monte Carlo precision rather than the definition of the test statistic itself.

---

# Extending the framework

The hypothesis-testing API is designed so that new procedures can reuse the common constructor and existing calibration engines.

A new hypothesis starts with

```julia
struct MyHypothesis <: CopulaHypothesis end
```

and then declares its name, null hypothesis, supported statistics, and calibrations:

```julia
Copulas.testname(::MyHypothesis) = "My copula hypothesis test"

Copulas.nullhypothesis(::MyHypothesis) = "The null hypothesis holds."

Copulas._available_statistics(::MyHypothesis) = (:Sn, :ks)

Copulas._available_calibrations(::MyHypothesis, ::Val{:Sn},) = (:simulation,)
```

The statistic is added through dispatch:

```julia
function Copulas._teststatistic(::MyHypothesis, ::Val{:Sn}, U; kwargs...,)
    # Compute and return the observed statistic.
end
```

If the generic simulation engine is appropriate, the hypothesis only needs to specify how to generate data under its null:

```julia
function Copulas._simulation_sample(::MyHypothesis, U, rng,)
    # Return a d × n sample generated under H₀.
end
```

The generic constructor then works automatically:

```julia
test = CopulaTest(MyHypothesis(), U; N=999,)
```

No change to `CopulaTest`, the generic result type, or the display machinery is required.

For a more complete description of the extension contract, see the [Developer Guide](@ref developer_fitting).

---

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
