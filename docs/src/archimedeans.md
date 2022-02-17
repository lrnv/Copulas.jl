# Archimedean Copulas

Details about what are archimedean copulas (on the math level)

# Generic Archimedean Copulas

Details about the generic construction of archimedean copulas (in the package), 
and details on exported methods that corresponds to this class 

```@docs
ArchimedeanCopula
```

# Implement your own 

Explain how easy it is to implement your own archimedean copulas and work with them. methods: 

```julia
struct MyAC{d,T} <: ArchimedeanCopula{d}
    par::T
end
ϕ(C::MyAC{d},x)
ϕ⁻¹(C::MyAC{d},x)
τ(C::MyAC{d})
τ⁻¹(::MyAC{d},τ)
radial_dist(C::MyAC{d})
```

# Available Archimedean copulas

## Independence

```@docs
IndependentCopula
```

## Clayton

```@docs
ClaytonCopula
```

## Franck

```@docs
FranckCopula
```

## Gumbel

```@docs
GumbelCopula
```

## Ali-Mikhail-Haq

```@docs
AMHCopula
```

## Joe

```@docs
JoeCopula
```