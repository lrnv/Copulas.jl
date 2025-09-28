```@meta
CurrentModule = Copulas
```

# [Archimax family](@id Archimax\_theory)

*Archimax copulas* form a hybrid family that combines an Archimedean generator $\phi$ with an extreme-value tail defined by its *stable tail dependence function* $\ell$, or its associated *Pickands function* $A_{\ell}(t) = \ell(\frac{t}{\lVert t \rVert})$. They interpolate between purely Archimedean and purely EV structures and underpin families such as **BB4** and **BB5**.

!!! note "Bivariate only (for now)"
    This section and the current implementation address the **bivariate case**. Multivariate extensions are possible; if you’d like to contribute, we’re happy to provide guidance on how to integrate them.

An Archimax copula [caperaa2000](@cite) $C$ admits the representation

$$C_{\phi,\ell}(u_1,u_2)=\phi\!\left(\ell\!\big(\phi^{-1}(u_1),\,\phi^{-1}(u_2)\big)\right), \qquad 0\le u_1,u_2\le 1,$$

where $\phi:[0,\infty)\to(0,1]$ is a given Archimedean generator (with inverse $\phi^{-1}$), $A_{\ell}:[0,1]\to[1/2,1]$ is a Pickands dependence function, and

$$\ell(x_1,...x_d)= \left(\sum_{i=1}^d x_i\right)\,A_{\ell}\!\left(\frac{x_i}{\left(\sum_{i=1}^d x_i\right)}, i \in 1,...,d\right),\qquad x_1,..,x_d\ge 0.$$

When $d = 2$, we abuse the $A$ notation by setting $A(w) = A(w, 1-w)$.

---

## Archimax in this package

This package provides the abstract type [`ArchimaxCopula`](@ref). Because we expose a wide set of Archimedean generators and extreme-value copulas, **many combinations are possible**: any Archimedean copula can be paired with any extreme-value copula to produce a valid Archimax copula.

The API is minimal and generic:

* Provide an Archimedean generator `gen::Generator` with methods `ϕ(G, s)` and `ϕ⁻¹(G, u)`. See [`Generator`](@ref) and [available Archimedean generators](@ref available_archimedean_models).
* Provide an extreme-value tail `tail::Tail` with its Pickands function `A(tail, w)` ro stable tail dependence function `ℓ(tail, x)`. See [`ExtremeValueCopula`](@ref) and [available extreme-value models](@ref available_extreme_models).

With these conventions, the constructor `ArchimaxCopula(d, gen, tail)` produces the correct d-variate model, accross all possiibilities through all implemented (and obviously new user-defined) models.

You can define an archimax copula as follows: 
```@example
using Copulas, Distributions, Plots
C = ArchimaxCopula(2, 
    Copulas.FrankGenerator(0.8),                   # Archimedean generator
    Copulas.AsymGalambosTail(0.35, 0.65, 0.3)    # Stable Tail Dependence
)
plot(C)
```

---

# Advanced Concepts

!!! note "Tail behaviour:"
    The **upper tail** is governed by the extreme value structure, while the **lower tail** is driven by the curvature of the Archimedean generator $\phi$. For specific families (e.g., BB5Copula) there are closed forms for $\lambda_U$ and for lower-tail orders.

!!! theorem "Theorem (Exhaustivity and consistency):" 
    For bivariate Archimax copulas,

    $$\tau_{\phi,A} \;=\; \tau_A \;+\; (1-\tau_A)\,\tau_\phi,$$

    where $\tau_A$ is Kendall’s $\tau$ of the EV copula with Pickands $A$, and $\tau_\phi$ is Kendall’s $\tau$ of the Archimedean copula with generator $\phi$ (Capéraà, Fougères & Genest, 2000).

---

### Classical constructions

* **BB4:** `GalambosTail` (EV) + `ClaytonGenerator` (only positive dependence suported yet) *gamma LT* (LT family **includes** Clayton as a special case).
* **BB5:** `GalambosTail` (EV) + *positive stable LT* (LT family **includes** Gumbel as a limiting case).

Each has its own docstring and dedicated section in this documentation.

## Simulation of Archimax Copulas

The implemented simulation scheme is the “frailty + EV” construction (e.g. [caperaa2000](@cite) [mai2012simulating](@cite), which is valid only when the Archimedean generator $\phi$ is **completely monotone**, which means it is the Laplace transform of a non-negative random variable $M$ called its *frailty*. If $(V_1,V_2)$ follows the EV copula with stable tail dependence function $\ell$, then

$$U_j \;=\; \phi\!\big(-\log V_j\,/\,M\big),\qquad j=1,2,$$

has the Archimax copula $C_{\phi,A}$.

!!! algorithm "Algorithm (Bivariate Archimax sampling):"

    * Simulate $(V_1,V_2) \sim C_{\text{EV}}$ with stable tail function $\ell$ (i.e., Pickands $A$).
    * Simulate a frailty $M \ge 0$ whose Laplace transform is $\mathbb{E}[e^{-sM}] = \phi(s)$.
    * Set $U_j := \phi\!\big(-\log(V_j)/M\big)$, $j=1,2$. Return $(U_1,U_2)$.

!!! todo "Allow any generator"
    According to [charpentier2014](@cite), it should be possible to use any d-monotonous generator. If you want to implement the corresponding sampler, please reach out.

**Notes on the objects used.**

* *Frailty distribution.* By Bernstein’s theorem, a completely monotone $\phi$ is a Laplace transform. The helper `frailty(G::Generator)` returns a distribution for $M$ such that `E(exp(-s*M)) = ϕ(gen, s)` and returns an error if the generator is nt completely monotonous.
* *EV sampling.* Step (1) uses the EV sampler already provided in this package (via `ℓ(tail::Tail, x)` and `A(tail::Tail, x)`), as documented in the EV section.
* *Generality.* The recipe extends to $d>2$ by simulating $(V_1,\dots,V_d)$ (But remember that it is not so simple to obtain it) from the EV copula of variable $d$ and applying steps (2)–(3) by components.

```@docs; canonical=false
ArchimaxCopula
```

## Conditionals and distortions

Let $C_{\phi,\ell}(\boldsymbol u)=\phi\!\big(\ell(\phi^{-1}(\boldsymbol u))\big)$ denote an Archimax copula (bivariate in our current implementation). For any copula, conditioning is given by partial-derivative ratios:

$$C_{I\mid J}(\boldsymbol u_I\mid \boldsymbol u_J)\;=\;\frac{\partial^{|J|}}{\partial \boldsymbol u_J}\,C(\boldsymbol u_I,\boldsymbol u_J)\,\bigg/\,\frac{\partial^{|J|}}{\partial \boldsymbol u_J}\,C(\boldsymbol 1_I,\boldsymbol u_J).$$

In the bivariate case ($d=2$) conditioning on $U_2=v$, the univariate conditional distortion reads

$$H_{1\mid 2}(u\mid v)\;=\;\frac{\partial}{\partial v}\,C_{\phi,\ell}(u,v)\,\bigg/\,\frac{\partial}{\partial v}\,C_{\phi,\ell}(1,v).$$

Using the chain rule and setting $s_1=\phi^{-1}(u)$, $s_2=\phi^{-1}(v)$, this becomes

$$H_{1\mid 2}(u\mid v)\;=\;\frac{\phi'\!\big(\ell(s_1,s_2)\big)\;\partial_2 \ell(s_1,s_2)\;\big(\phi^{-1}\big)'(v)}{\phi'\!\big(\ell(0,s_2)\big)\;\partial_2 \ell(0,s_2)\;\big(\phi^{-1}\big)'(v)} \;=\;\frac{\phi'\!\big(\ell(s_1,s_2)\big)\;\partial_2 \ell(s_1,s_2)}{\phi'\!\big(\ell(0,s_2)\big)\;\partial_2 \ell(0,s_2)},$$

where the factor $(\phi^{-1})'(v)$ cancels. When $\ell$ comes from a Pickands function $A$ (bivariate EV case), $\partial_2 \ell$ is available in closed form. This is the expression used by the implementation for conditional distortions on the copula scale; higher-dimensional extensions follow the same principle with higher-order partial derivatives.

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
