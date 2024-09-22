```@meta
CurrentModule = Copulas
```

# Liouville Copulas

!!! todo "Not merged yet !"
    Liouville copulas are coming in this PR : https://github.com/lrnv/Copulas.jl/pull/83, but the work is not finished. 

Archimedean copulas have been widely used in the literature due to their nice decomposition properties and easy parametrization. The interested reader can refer to the extensive literature [hofert2010,hofert2013a,mcneil2010,cossette2017,cossette2018,genest2011a,dibernardino2013a,dibernardino2013a,dibernardino2016,cooray2018,spreeuw2014](@cite) on Archimedean copulas, their nesting extensions and most importantly their estimation. 

One major drawback of the Archimedean family is that these copulas have exchangeable marginals (i.e., $C(\bm u) = C(\mathrm{p}(\bm u))$ for any permutation $p(\bm u)$ of $u_1,...,u_d$): the dependence structure is symmetric, which might not be a wanted property. However, from the Radial-simplex expression, we can easily extrapolate a little and take for $\bm S$ a non-uniform distribution on the simplex. 

Liouville's copulas share many properties with Archimedean copulas, but are not exchangeable anymore. This is an easy way to produce non-exchangeable dependence structures. See [cote2019](@cite) for a practical use of this property.

Note that Dirichlet distributions are constructed as $\bm S = \frac{\bm G}{\langle \bm 1, \bm G\rangle}$, where $\bm G$ is a vector of independent Gamma distributions with unit scale (and potentially different shapes: taking all shapes equal yields the Archimedean case). 

```@bibliography
Pages = ["Liouville.md"]
Canonical = false
```