```@meta
CurrentModule = Copulas
```

# [Archimax Copulas](@id Archimax\_theory)

*Archimax copulas* form a hybrid family that combines an Archimedean copula (via its generator) with an extreme-value copula (via its Pickands function). In dimension 2, they interpolate between purely Archimedean and purely EV structures and underpin families such as **BB4** and **BB5**.

!!! note "Bivariate only (for now)"
This section and the current implementation address the **bivariate case**. Multivariate extensions are possible; if you’d like to contribute, we’re happy to provide guidance on how to integrate them.

A bivariate Archimax copula [caperaa2000bivariate](@cite) $C$ admits the representation

$$
C_{\phi,A}(u_1,u_2)
=\phi\!\left(\ell_A\!\big(\phi^{-1}(u_1),\,\phi^{-1}(u_2)\big)\right),
\qquad 0\le u_1,u_2\le 1,
$$

where $\phi:[0,\infty)\to(0,1]$ is an Archimedean generator (with inverse $\phi^{-1}$), $A:[0,1]\to[1/2,1]$ is a Pickands dependence function, and

$$
\ell_A(x,y)=(x+y)\,A\!\left(\frac{x}{x+y}\right),\qquad x,y\ge 0.
$$

---

## Archimax in this package

This package provides the abstract type [`ArchimaxCopula`](@ref) as a foundation for **bivariate** Archimax copulas. Because we expose a wide set of Archimedean generators and extreme-value copulas, **many combinations are possible**: any Archimedean copula can be paired with any extreme-value copula to produce a valid Archimax copula.

The API is minimal and generic:

* Provide an Archimedean generator `G<:Generator` with methods `ϕ(G, s)` and `ϕ⁻¹(G, u)`. See [`Generator`](@ref) and [available Archimedean generators](@ref available_archimedean_models).
* Provide an extreme-value copula `E<:ExtremeValueCopula` with its Pickands function `A(E, t)`. See [`ExtremeValueCopula`](@ref) and [available extreme-value models](@ref available_extreme_models).

With these conventions, the constructor `ArchimaxCopula(gen::G, evd::ExtremeValueCopula)` uses the above representation directly and works uniformly across all implemented (and obviously new user-defined) models.

---

# Advanced Concepts

* **Tail behaviour.**
  The **upper tail** is governed by the Pickands function $A$ (EV structure), while the **lower tail** is driven by the curvature of the Archimedean generator $\phi$. For specific families (e.g., BB5) there are closed forms for $\lambda_U$ and for lower-tail orders.

* **Kendall’s tau (theorem).**
  For bivariate Archimax copulas,

  $$
  \tau_{\phi,A} \;=\; \tau_A \;+\; (1-\tau_A)\,\tau_\phi,
  $$

  where $\tau_A$ is Kendall’s $\tau$ of the EV copula with Pickands $A$, and $\tau_\phi$ is Kendall’s $\tau$ of the Archimedean copula with generator $\phi$ (Capéraà, Fougères & Genest, 2000).

---

### Classical constructions

* **BB4:** Galambos (EV) + *gamma LT* (LT family **includes** Clayton as a special case).
* **BB5:** Galambos (EV) + *positive stable LT* (LT family **includes** Gumbel as a limiting case).

Each has its own docstring and dedicated section in this documentation.

## Simulation of Archimax Copulas

For Archimax copulas we follow the “frailty + EV” construction (e.g. Capéraà–Fougères–Genest, 2000; Mai-Scherer, 2012). When the Archimedean generator $\phi$ is **completely monotone**, it is the Laplace transform of a non-negative random variable $M$ (frailty). If $(V_1,V_2)$ follows the EV copula with stable tail dependence function $\ell$ (equivalently Pickands $A$), then

$$
U_j \;=\; \phi\!\big(-\log V_j\,/\,M\big),\qquad j=1,2,
$$

has the Archimax copula $C_{\phi,A}$.

!!! algorithm "Algorithm (Bivariate Archimax sampling):"

* Simulate $(V_1,V_2) \sim C_{\text{EV}}$ with stable tail function $\ell$ (i.e., Pickands $A$).
* Simulate a frailty $M \ge 0$ whose Laplace transform is $\mathbb{E}[e^{-sM}] = \phi(s)$.
* Set $U_j := \phi\!\big(-\log(V_j)/M\big)$, $j=1,2$. Return $(U_1,U_2)$.

**Notes on the objects used.**

* *Frailty distribution.* By Bernstein’s theorem, a completely monotone $\phi$ is a Laplace transform. The helper `frailty(gen)` returns a distribution for $M$ such that `E(exp(-s*M)) = ϕ(gen, s)`.
* *EV sampling.* Step (1) uses the EV sampler already provided in this package (via the Pickands function `A(evd, t)`), as documented in the EV section.
* *Generality.* The recipe extends to $d>2$ by simulating $(V_1,\dots,V_d)$ (But remember that it is not so simple to obtain it) from the EV copula of variable $d$ and applying steps (2)–(3) by components.

```@docs
ArchimaxCopula
```

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
