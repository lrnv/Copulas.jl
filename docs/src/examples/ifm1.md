# Influence of the method of estimation

The ability to fit the copula and marginals directly through the `SklarDist` interface is very practical for quickly fitting a model to a dataset. It works as follows: 
```@example ifm
using Copulas, Distributions
X₁ = Normal()
X₂ = LogNormal()
X₃ = Gamma()
C = GaussianCopula([
    1.0 0.4 0.1
    0.4 1.0 0.8
    0.1 0.8 1.0
])
D = SklarDist(C,(X₁,X₂,X₃))
x = rand(D,100)
```

And we can fit the same model directly as follows: 
```@example ifm
quick_fit = fit(SklarDist{GaussianCopula, Tuple{Normal,LogNormal,Gamma}}, x)
```

However, we should clarify what this is doing. There are several ways of estimating compound parametric models: 

- **Joint estimation (JMLE):** This method simply computes the joint log-likelihood $\log f_D$ of the random vector `D` and maximizes, jointly, with respect to the parameters of the dependence structure and the marginals. This is the easiest to understand, but not the easiest numerically, as the resulting log-likelihood can be highly non-linear and computations can be tedious.
- **Inference functions for margins (IFM):** This method splits the process into two parts: the marginal distributions $F_{i}, i \in 1,..d$ are estimated first, separately, through maximum likelihood on marginal data. Denote $\hat{F}_{i}$ these estimators. Then, we fit the copula on pseudo-observations, which can be computed in two ways: 
    - **IFM1:** Use empirical ranks to compute the pseudo-observations.
    - **IFM2:** Use the estimated distribution functions $\hat{F}_{i}$ for the marginals to compute the pseudo-observations as $u_{i,j} = \hat{F}_{i}(x_{i,j})$.

The `fit(SklarDist{...},...)` method in `Copulas.jl` is implemented as follows: 

```julia
function Distributions.fit(::Type{SklarDist{CT,TplMargins}},x) where {CT,TplMargins}
    # The first thing to do is to fit the marginals : 
    @assert length(TplMargins.parameters) == size(x,1)
    m = Tuple(Distributions.fit(TplMargins.parameters[i],x[i,:]) for i in 1:size(x,1))
    u = pseudos(x)
    C = Distributions.fit(CT,u)
    return SklarDist(C,m)
end
```

This clearly performs **IFM1** estimation. **IFM2** is not much harder to implement, and could be done for our model as follows:
```@example ifm
# Marginal fits are the same than IFM1, so we just used those to compute IFM2 ranks:
u = similar(x)
for i in 1:length(C)
    u[i,:] .= cdf.(Ref(quick_fit.m[i]), x[i,:])
end

# estimate a Gaussian copula: 
ifm2_cop = fit(GaussianCopula,u)
```

Let us compare the two obtained correlation matrices: 
```@example ifm
ifm2_cop.Σ .- quick_fit.C.Σ
```

We see that the estimated parameter is not exactly the same, which is normal. Even in this contrived example, the difference between the two is not striking. Whether one method is better than the other is unclear, but the JMLE method is clearly superior by definition. However, due to its complexity, most software do not perform such estimation.