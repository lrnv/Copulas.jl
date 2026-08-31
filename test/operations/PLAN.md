# Operation-oriented test migration

The test suite currently organizes the four proof obligations by proof kind:
contracts, mathematical correctness, specialization equivalence, and routing
exhaustiveness. This makes a public operation span several files. The target
organization groups those obligations by public operation instead.

## Target layout

- `constructors.jl`: constructor forms, inference, and validation.
- `distribution.jl`: `cdf`, `logcdf`, `pdf`, `logpdf`, and likelihoods.
- `sampling.jl`: `rand` and `rand!`.
- `subsetting.jl`: `subsetdims`.
- `conditioning.jl`: marginal distortions and conditional joint laws.
- `rosenblatt.jl`: direct and inverse Rosenblatt transforms.
- `measure.jl`: rectangle probabilities.
- `dependence.jl`: scalar and pairwise dependence measures and inverses.
- `fitting.jl`: fitting discovery, execution, and model-result API.
- `generators.jl`, `tails.jl`, and `univariate_distributions.jl`: public
  component APIs.
- `sklar.jl`, `nataf.jl`, and `utilities.jl`: standalone public compositions.

Cross-operation identities, family-specific mathematical facts, numerical
regressions, and extensions remain separate; they must not be forced into an
operation merely for uniformity.

## Required proof inside an operation file

Each operation must visibly establish all applicable obligations:

1. Apply its inexpensive public contract to every canonical family in
   `COPULA_FIXTURES`.
2. Validate the generic implementation against an independent mathematical
   oracle.
3. Discover specialized implementations from `ALL_COPULA_CASES` and compare
   them with the generic definition whenever that fallback is a valid oracle.
4. Close local `selected_routes` and `tested_routes` sets by equality.

Value-dependent branches are invisible to `which`; every relevant regime must
therefore have a representative in `ALL_COPULA_CASES`. Route deduplication must
not discard representatives intentionally added for such branches.

## Migration sequence

- [x] Migrate `measure` and `subsetdims` as initial deterministic examples.
- [x] Extract Rosenblatt and inverse Rosenblatt.
- [ ] Finish scalar and pairwise dependence measures (the contract and route
      execution are migrated; mathematical oracles remain to consolidate).
- [ ] Finish distribution evaluation (the family-wide contract is migrated;
      generic and specialized proofs remain to consolidate).
- [x] Extract sampling.
- [ ] Finish conditioning and its distortion API (the family-wide contract is
      migrated; generic and specialized proofs remain to consolidate).
- [ ] Extract fitting and model-result behavior.
- [ ] Consolidate generator, tail, and auxiliary univariate APIs.
- [ ] Move Sklar, Nataf, constructors, and standalone utilities.
- [ ] Remove the old obligation directories once no tests remain in them.
- [ ] Replace this plan with the stable developer-guide instructions.

Migration is assertion-preserving: an old assertion is removed only after its
contract, oracle, equivalence, or regression role has been retained in the new
location.
