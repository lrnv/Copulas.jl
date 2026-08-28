# Test architecture

The suite validates a public behaviour through four complementary obligations.
Passing only one of them is not sufficient.

1. **Contract coverage.** Every public family is constructed and the observable
   contract is exercised on it. This proves that the operation is available and
   returns values with the documented shape, support, bounds, and type.
2. **Generic correctness.** Each generic implementation mechanism is checked
   against an independent mathematical oracle. Expensive identities are tested
   once per mechanism, not once per family.
3. **Specialization equivalence.** Every deterministic specialization that
   replaces a generic implementation is compared with that generic path at an
   ordinary interior point. If the generic path is not mathematically applicable,
   the specialization must instead be checked against an independent identity.
4. **Route exhaustiveness.** A registry or dispatch inventory proves that every
   public family reaches either a validated generic mechanism or one of the
   validated specializations.

Together, these obligations establish the intended implication

```text
correct generic mechanisms
+ equivalent (or independently correct) specializations
+ every family routed through one of those mechanisms
= correct public behaviour for every family.
```

## Layout

- `obligations/contracts/` implements obligation 1 and maintains exhaustive
  public-family and public-symbol registries.
- `obligations/correctness/` implements obligation 2 with independent
  mathematical and statistical oracles.
- `obligations/equivalence/` implements obligation 3. A specialization belongs
  here only when it is compared with a fallback or an independent identity.
- `obligations/routing/` implements obligation 4 by discovering and exercising
  every distinct method selected by the public fixtures. Merely executing a
  method establishes routing, not correctness or equivalence.
- Statistical tests replace draw-by-draw equivalence for random samplers with
  distributional identities.
- `Aqua.jl` and `fixtures.jl` provide infrastructure shared by all obligations.
- `families/` contains parameter boundaries, singular atoms, published values,
  and regressions that cannot be derived from the shared contracts.
- `extensions/` contains contracts and regressions for optional package
  extensions.

## Behaviour checklist

Each public behaviour must be accounted for as follows.

| Behaviour | Contract | Generic oracle | Specialized paths | Exhaustive routing |
|:--|:--|:--|:--|:--|
| construction and validation | every public family | canonical `{d}` constructor | reductions and inferred forms | constructor registry |
| CDF, log-CDF, PDF and log-PDF | every applicable family | derivatives and numerical integration | deterministic formulas vs fallback | dispatch inventory |
| sampling | every public family | distributional identities | no draw-by-draw comparison | sampler dispatch inventory |
| subsetting | every public family | marginal CDF identity | specialized subsets vs parent | dispatch inventory |
| conditioning | every public family | normalized mixed derivatives | distortions vs generic conditional | distortion and dispatch registries |
| Rosenblatt transforms | every public family | conditional-CDF factorization | specialized transforms vs generic | dispatch inventory |
| dependence measures | applicability on every family | defining integral or statistical identity | closed forms vs generic/independent oracle | one execution per dispatch |
| fitting | every advertised family/method | recovery and parameter-map identities | specialized estimators vs their defining statistic | advertised-method registry |
| generator primitives | every numerical public generator; explicit reduction contract for marker generators | differentiation and inversion identities | closed forms vs generic primitive | generator registry |
| tail primitives | every public tail | homogeneity, convexity, and derivative identities | analytic partials vs AD/finite differences | tail registry |
| Sklar composition | public composition contract | change-of-variable identities | specialized conditioning/transforms vs generic | composition paths |
| optional extensions | every declared extension | extension-specific public identity | extension-specific | extension registry |

When adding a public family or a specialized method, update the corresponding
registry and supply the missing proof obligation. Tests should not repeat an
expensive mathematical identity for every family merely to obtain coverage.
