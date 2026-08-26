# Redesign de l’architecture des tests

Les issues #422, #424, #425 et #430 pointent déjà dans la bonne direction, mais le design doit être précisé davantage.

Aujourd’hui, le problème principal est visible dans quatre fichiers :

- `GenericTests.jl` : 930 lignes ;
- `ExtremeValueArchitecture.jl` : 1 449 lignes ;
- `NestedArchimedeanCopula.jl` : 723 lignes ;
- `ConditionalDistribution.jl` : 701 lignes.

`GenericTests.jl` mélange actuellement :

- contrat public ;
- détection des capacités ;
- introspection du dispatch ;
- tests statistiques ;
- intégration numérique ;
- propriétés mathématiques propres à certaines familles ;
- exemptions ad hoc.

Le résultat est une matrice implicite « toutes les copules × presque toutes les opérations », avec beaucoup de spécialisations Julia compilées uniquement pour répéter la même propriété.

## Architecture proposée

### 1. Contrats de l’API publique

Créer un helper court par opération publique :

```julia
test_core_api(C)
test_density_api(C)
test_sampling_api(C)
test_subsetting_api(C)
test_conditioning_api(C)
test_rosenblatt_api(C)
test_fitting_api(CT, data; method)
test_dependence_api(C)
```

Chaque helper vérifie uniquement le contrat public :

- formes et types des résultats ;
- support et frontières ;
- erreurs attendues ;
- invariants simples ;
- aller-retour lorsqu’il est mathématiquement garanti.

Par exemple, `test_conditioning_api` vérifierait une seule fois :

```julia
D = condition(C, j, v)
minimum(D) == 0
maximum(D) == 1
cdf(D, 0) == 0
cdf(D, 1) == 1
quantile(D, p) ∈ [0, 1]
```

Il ne vérifierait ni une formule Clayton particulière, ni une comparaison AD, ni une intégration numérique.

### 2. Contrats des composants internes

Au lieu d’assembler chaque générateur avec plusieurs dimensions et de refaire toute l’API d’une copule, tester directement les composants :

```text
contracts/
  generators.jl
  tails.jl
  distortions.jl
  radials.jl
  samplers.jl
```

Pour chaque générateur :

- `ϕ`, son inverse et ses dérivées ;
- monotonie ;
- frontières ;
- méthodes fermées propres à la famille.

Pour chaque tail EV :

- homogénéité de `ℓ` ;
- marges ;
- Pickands lorsque disponible ;
- dérivées partielles ;
- représentation spectrale éventuelle.

Pour chaque distortion :

- CDF monotone ;
- quantile généralisé ;
- support ;
- atomes éventuels ;
- cohérence avec une référence analytique.

Ensuite, seuls quelques modèles assemblés vérifient que le frontend `Copula` relie correctement ces composants.

Cela évite par exemple de compiler tout `condition + Rosenblatt + pdf + intégration` pour chaque générateur archimédien alors que ces familles partagent la même infrastructure.

### 3. Tests par chemin de dispatch

Créer un registre central, explicite et lisible :

```julia
const PATH_CASES = (
    generic_cdf          = SomeCopula(...),
    generic_density      = SomeCopula(...),
    matrix_sampler       = ClaytonCopula{5}(...),
    frailty_sampler      = FrankCopula{3}(...),
    biv_ev_distortion    = GalambosCopula{2}(...),
    generic_condition    = RafteryCopula{2}(...),
    singular_condition   = MCopula{2}(),
    numerical_ev         = HuslerReissCopula{3}(...),
    fractional_williamson = LiouvilleCopula{2}(...),
)
```

Chaque mécanisme n’a besoin que d’un ou deux représentants.

Le registre doit remplacer les dizaines de prédicats actuels :

```julia
can_pdf(C)
can_ad(C)
check_rosenblatt(C)
check_corkendall(C)
can_integrate_pdf(C)
check_biv_conditioning(C)
```

Ces prédicats reconstituent actuellement une API de capacités parallèle, uniquement dans les tests, et deviennent rapidement faux.

### 4. Régressions propres aux familles

Les fichiers familiaux ne conservent que ce qui distingue réellement la famille :

- constructeurs et paramètres invalides ;
- valeurs de référence publiées ;
- formes fermées ;
- limites particulières ;
- bugs numériques déjà rencontrés ;
- algorithmes d’échantillonnage particuliers.

Aucun test générique de forme, support ou conditionnement ne doit y être recopié.

## Structure de fichiers proposée

```text
test/
  runtests.jl
  fixtures.jl

  contracts/
    core.jl
    density.jl
    sampling.jl
    subsetting.jl
    conditioning.jl
    transforms.jl
    fitting.jl
    dependence.jl

  components/
    generators.jl
    tails.jl
    distortions.jl
    radial_distributions.jl

  paths/
    dispatch_paths.jl
    numerical_paths.jl
    integration_paths.jl

  families/
    archimedean.jl
    elliptical.jl
    extreme_value.jl
    liouville.jl
    miscellaneous.jl
    nested.jl

  extensions/
    expectation_maximization.jl
```

Pas de macro compliquée. De simples fonctions de test et des tuples de fixtures suffisent.

## Couvrir « tous les cas possibles »

Ne pas interpréter cela comme le produit cartésien de tous les axes. Inventorier les axes indépendants :

- dimension 2, 3 et dimension supérieure ;
- `Float32`, `Float64`, `BigFloat` lorsque promis ;
- modèle régulier, singulier et mixte ;
- paramètres intérieurs et cas limites ;
- formule fermée et fallback numérique ;
- générateur à frailty continue, discrète ou Williamson générique ;
- ordre entier et fractionnaire ;
- sampler direct, frailty, spectral ou générique ;
- conditionnement spécialisé et fallback ;
- CDF analytique, quadrature et noyau probabiliste.

Sélectionner ensuite un petit ensemble couvrant tous ces axes et chemins, sans tester toutes leurs combinaisons.

La matrice de couverture devrait être une donnée Julia lisible, pas un document séparé susceptible de devenir obsolète.

## Réduction des tests coûteux

Supprimer du bestiaire global :

- l’intégration de chaque densité pour chaque modèle ;
- les comparaisons échantillonnage/CDF pour chaque famille ;
- `corkendall` sur chaque modèle ;
- Rosenblatt sur toutes les variantes paramétriques ;
- les comparaisons systématiques fast path/fallback ;
- les boucles sur de nombreux points lorsqu’un seul point compile exactement le même chemin.

À la place :

- une intégration de densité par mécanisme ;
- une validation statistique par sampler ;
- un aller-retour Rosenblatt par conditionneur ;
- des tests analytiques ciblés pour les familles singulières ;
- deux ou trois points seulement pour les frontières et l’intérieur.

Réduire `rand(C, 1000)` à `rand(C, 10)` ne changera presque rien lorsque la compilation domine. En revanche, ne jamais appeler une combinaison composite redondante évitera entièrement sa compilation.

## Stratégie CI

Conserver un seul processus par version Julia afin de partager la compilation entre fichiers.

- Julia LTS : suite complète.
- Julia stable : contrats publics, constructeurs et principaux chemins de dispatch.
- Tests numériques lourds : LTS uniquement.
- Benchmarks : workflow séparé, comme actuellement.
- Extensions : jobs séparés uniquement si elles nécessitent des environnements distincts.

Éclater chaque famille dans un job parallèle réduirait peut-être le temps mural, mais recompilierait Copulas.jl et les dépendances dans chaque job. Ce serait probablement un mauvais échange.

## Ordre d’implémentation

1. Ajouter un chronométrage par fichier et établir la baseline de #425.
2. Créer `fixtures.jl` et les helpers de contrats sans supprimer de tests.
3. Construire la matrice des chemins de dispatch demandée par #424.
4. Migrer progressivement `GenericTests.jl`.
5. Supprimer les duplications des fichiers familiaux.
6. Réorganiser les fichiers seulement après stabilisation du contenu.
7. Comparer temps total, temps de compilation et nombre de `MethodInstance`.

Le premier objectif concret est de supprimer entièrement `GenericTests.jl`, remplacé par des contrats courts et un registre de chemins explicite. C’est le meilleur point d’entrée : il clarifie simultanément l’API, la couverture et la source du coût de compilation.
