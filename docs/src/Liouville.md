```@meta
CurrentModule = Copulas
```

## Liouville Copulas

Archimedean copulas have been widely used in the literature due to their nice decomposition properties and easy parametrization. The interested reader can refer to the extensive literature [hofert2010,hofert2013a,mcneil2010,cossette2017,cossette2018,genest2011a,dibernardino2013a,dibernardino2013a,dibernardino2016,cooray2018,spreeuw2014](@cite) on Archimedean copulas, their nesting extensions and most importantly their estimation. One major drawback of the Archimedean family is that these copulas have exchangeable marginals (i.e., $C(\bm u) = C(\mathrm{p}(\bm u))$ for any permutation $p(\bm u)$ of $u_1,...,u_d$): the dependence structure is symmetric, which might not be a wanted property. However, from the Radial-simplex expression, we can easily extrapolate a little and take for $\bm S$ a non-uniform distribution on the simplex. 

**Definition (Liouville Copulas):** For $R$ a positive random variable, $\bm\alpha \in N^d$ and $S \sim \mathrm{Dirichlet}(\bm \alpha)$ a Dirichlet random vector on the simplex, the copula of the random vector $R\bm S$ is called the Liouville copula with radial part $R$ and Simplex parameters $\alpha$. 

Liouville's copulas share many properties with Archimedean copulas, but are not exchangeable anymore. This is an easy way to produce non-exchangeable dependence structures. See [cote2019](@cite) for a practical use of this property.

Note that Dirichlet distributions are constructed as $\bm S = \frac{\bm G}{\langle \bm 1, \bm G\rangle}$, where $\bm G$ is a vector of independent Gamma distributions with unit scale (and potentially different shapes: taking all shapes equal yields the Archimedean case). 

There are still a few properties that are interesting for the implementation, but wich requires a few notations. Let's denote by 

$$\phi_{d}$$

the inverse Williamson $d$-transform of $R$ for any integer $d$.

Then the distribution function of the Liouville copula is given by 

$$C(\bm u) = \sum_{\bm i \le \bm \alpha} \frac{(-1)^{\lvert\bm i\rvert}}{\bm i!} * \phi^{(\lvert\bm i\rvert)}(\lvert \bm x \rvert) * \bm x^{\bm i},$$

where $x_i = F_{\phi_{\alpha_i}}^{-1}(u_i)$ for all i.

And for sampling, the same kind of algorithm is availiable. 

!!! note "Complexity of the matter"
    Note that here we need access to the quantile functions of all the (inverse) Williamson $\alpha_i$-transform of $\phi_{\lvert\bm\alpha\rvert}$, which itself is the $\lvert\bm\alpha\rvert$-Williamson transfrom of $R$. This complexity adds up quickly, but the package internally uses fast Faa-di-bruno formula at compile time to overcome much of it. 

You can sample and compute the cdf and pdf of *any* Liouville copula, even one for which you provide the generator and/or the Radial distribution yourself.


```@docs
LiouvilleCopula
```


```@bibliography
Pages = ["Liouville.md"]
Canonical = false
```