```@meta
CurrentModule = Copulas
```

# [Extreme Value family](@id Extreme_theory)

*Extreme value copulas* are fundamental in the study of rare and extreme events due to their ability to model dependency in situations of extreme risk. This package provides a wide selection of bivariate extreme value copulas; multivariate cases are not yet implemented. Feel free to open an issue or propose a pull request if you want to contribute a multivariate case. 

!!! info "Only Bivariate"
    The implementation here only deals with bivariate extreme value copulas. Multivariate cases are more tedious to implement, but not impossible: if you want to propose an implementation, we can provide guidance on how to merge it here. Do not hesitate to reach us on GitHub.

A bivariate extreme value copula [gudendorf2010extreme](@cite) $C$ has the following characteristic property:

$$C(u_1^t, u_2^t) = C(u_1, u_2)^t, \; t > 0.$$

It can be represented through its stable tail dependence function $\ell(\cdot)$:

$$C(u_1, u_2) = \exp\{-\ell(\log(u_1), \log(u_2))\},$$

or through a convex function $A: [0,1] \to [1/2, 1]$ satisfying $\max(t, t-1)\leq A(t) \leq 1,$ called its Pickands dependence function:

$$C(u_1,u_2)=\exp\left\{\log(u_1u_2)A\left(\frac{\log(u_1)}{\log(u_1u_2)}\right)\right\},$$

In the context of bivariate extreme value copulas, the functions $\ell$ and $A$ are related as follows:

$$\ell(u_1, u_2) = (u_1 + u_2)A\left(\frac{u_1}{u_1 + u_2}\right).$$

!!! tip "Only `A` is needed"
    In our implementation, it is sufficient to provide the Pickands dependence function $A$ to construct the extreme value copula and have it work correctly. Providing the other functions would, of course, improve performance.

In this package, there is an abstract type [`ExtremeValueCopula`](@ref) that provides a foundation for defining bivariate extreme value copulas. Many extreme value copulas are already implemented for you! See [this list](@ref available_extreme_models) to get an overview.

If you do not find the one you need, you may define it yourself by subtyping `ExtremeValueCopula`. The API requires only a method for the Pickands function `A(C::ExtremeValueCopula) = ...`. By providing this function, you can easily create a new extreme value copula that fits your specific needs:

```julia
struct MyExtremeValueCopula{P} <: ExtremeValueCopula{P}
    θ::P
end

A(C::ExtremeValueCopula, t) = (t^C.θ + (1 - t)^C.θ)^(1/C.θ) # This is the Pickands function of the Logistic (Gumbel) Copula
```

!!! info "Empirical EV estimator available"
    When you have bivariate data and want a nonparametric Extreme Value copula, you can estimate the Pickands function from pseudo-observations using `EmpiricalEVTail` and plug it into an `ExtremeValueCopula`. For convenience, `EmpiricalEVCopula(u)` builds the EV copula in one step. See the bestiary entry for [`EmpiricalEVTail`](@ref) and [`ExtremeValueCopula`](@ref) docs for details.

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

$$g_Z(z) = \frac{d}{dz} G_Z(z) = 1 + (1 - z)^{-1} \left(A(z) - z A'(z)\right)$$

The conditional distribution of $W$ given $Z$ is:

$$F(w|z) = \frac{1}{g_Z(z)} \frac{d}{dz} F(z, w),$$ 

which simplifies to:

$$F(w|z) = w \frac{z(1 - z) A'(z)}{A(z) g_Z(z)} + (w - w \log w) \left(1 - \frac{z(1 - z) A''(z)}{A(z) g_Z(z)} \right)$$

Given $Z$, the distribution of $W$ is uniform on $(0, 1)$ with probability $p(Z)$ and equals the product of two independent uniforms on $(0, 1)$ with probability $1 - p(Z)$, where:

$$p(z) = \frac{z(1 - z) A'(z)}{A(z) g_Z(z)}$$

Since $g_Z(z)$ is the derivative of the cumulative distribution function of $Z$, it holds that $0 \leq p(z) \leq 1$.

For the class of Extreme Value Copulas, We follow the methodology proposed by Ghoudi,1998. page 191. [ghoudi1998proprietes](@cite). Here, is a detailed algorithm for sampling from bivariate Extreme Value Copulas:

!!! algorithm "Bivariate Extreme Value Copulas sampling"

    * Simulate $U_1, U_2 \sim \mathcal{U}[0, 1]$
    * Simulate $Z \sim G_Z(z)$
    * Select $W = U_1$ with probability $p(Z)$ and $W = U_1U_2$ with probability $1 - p(Z)$
    * Return $X = W^{Z/A(Z)}$ and $Y = W^{(1 - Z)/A(Z)}$  

Note that all functions present in the algorithm were previously defined to ensure that the implemented methodology has a solid theoretical basis.

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

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```