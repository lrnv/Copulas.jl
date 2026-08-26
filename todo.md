# Redesign de l’architecture des tests

Les issues #422, #424, #425, #426, #428 et #430 pointent déjà dans la bonne direction, mais le design doit être précisé davantage.

Aujourd’hui, `GenericTests.jl` mélange contrat public, détection de capacités, introspection du dispatch, tests statistiques, intégration numérique, propriétés propres aux familles et exemptions ad hoc. Le résultat est une matrice implicite « toutes les copules × presque toutes les opérations », avec beaucoup de spécialisations Julia compilées uniquement pour répéter la même propriété.

## Principe directeur

Chaque copule doit être soumise au contrat public complet du paquet :

- formats de constructeurs ;
- interface de `Distributions.jl` ;
- sous-ensembles ;
- conditionnement ;
- transformations de Rosenblatt ;
- mesures de dépendance ;
- ajustement ;
- toute autre opération publiquement promise.

Le découpage en helpers sert seulement à rendre le code et les échecs lisibles. Il ne doit pas permettre de choisir opportunément quelles parties de l’API tester pour une famille donnée.

Les seules adaptations admises correspondent à des limites mathématiques explicites du contrat, par exemple l’absence de densité ordinaire ou de bijection de Rosenblatt pour certaines lois singulières ou mixtes. Ces adaptations doivent être documentées publiquement et testées comme telles, pas encodées dans une collection de prédicats ad hoc.

## 1. Établir une source de vérité pour l’API publique

La convention Julia/Pkg définit l’API publique par les comportements documentés des symboles publics. Un symbole est public s’il est déclaré avec `export`, ou avec `public` sans être injecté par `using`. Les déclarations de visibilité et la documentation comportementale sont donc toutes les deux normatives : aucune ne suffit seule.

Copulas.jl ne possède pas encore une représentation complète et cohérente de ce contrat :

- les `export` déclarent bien des symboles publics, mais ne décrivent pas leurs garanties ;
- certains symboles utilisés avec qualification, notamment les mesures de dépendance, sont documentés comme une API sans être exportés ni explicitement déclarés `public` ;
- les méthodes ajoutées à des fonctions publiques de `Distributions.jl`, `StatsBase.jl` ou d’autres dépendances ne figurent pas naturellement dans la liste des exports de Copulas.jl ;
- `docs/src/api/public.md` utilise `@autodocs Private=false` et fournit un inventaire automatique, pas un contrat comportemental ;
- le guide développeur contient la description la plus proche d’un contrat, mais sa table est annoncée comme non exhaustive et classe encore plusieurs opérations génériques comme optionnelles ;
- le manuel et les exemples promettent des comportements supplémentaires sans les rassembler au même endroit.

La source de vérité doit donc avoir trois couches cohérentes :

1. `export` et `public` déclarent exhaustivement les symboles appartenant à l’API de Copulas.jl ;
2. une table normative dans la documentation publique décrit les comportements promis ;
3. les contrats de test vérifient ces comportements.

Le guide développeur explique seulement les points d’extension internes permettant de satisfaire ce contrat.

Comme `public` n’existe nativement qu’à partir de Julia 1.11, relever la version minimale de Julia à 1.11 au moment de stabiliser cette API. Modifier ensemble `Project.toml`, les environnements de documentation et la matrice CI, puis utiliser directement le mot-clé `public` sans couche de compatibilité. Cette rupture de compatibilité Julia doit être annoncée et accompagnée du changement de version approprié selon la convention SemVer de Pkg.

Le job CI `lts` actuel doit alors disparaître ou être remplacé par un job explicite `1.11` tant que la LTS officielle est antérieure à la compatibilité minimale du paquet. Ne pas conserver un job symbolique qui sélectionne une version devenue incompatible. Les workflows de documentation, benchmarks, extensions et release doivent également être audités pour éviter qu’un sélecteur générique ou un ancien manifeste continue à tester une version hors contrat.

### Audit de visibilité

Construire trois inventaires comparables :

- symboles déclarés par `export` ou `public` ;
- fonctions et types présentés comme publics dans toute la documentation ;
- méthodes publiques ajoutées aux interfaces de dépendances.

Pour chaque différence, prendre une décision explicite :

- déclarer le symbole `public` s’il fait partie de l’API mais doit rester qualifié ;
- l’exporter s’il doit aussi être disponible après `using Copulas` ;
- le retirer ou le présenter comme interne dans la documentation s’il n’est pas promis ;
- documenter les méthodes étendant une fonction externe dans la table de l’API de Copulas.jl.

Les premiers candidats à examiner sont :

- les mesures `τ`, `ρ`, `β`, `γ`, `ι`, `λₗ` et `λᵤ`, ainsi que leurs variantes pairwise ;
- `measure` et `corkendall` ;
- les abstractions et fonctions de générateurs utilisées par les utilisateurs avancés ;
- les types de tails et représentations spectrales constructibles publiquement ;
- les fonctions de fitting et résultats venant de `Distributions.jl` et `StatsBase.jl` ;
- toute fonction qualifiée `Copulas.foo` dans le manuel ou les exemples.

Cette liste est un point de départ : l’audit doit être mécanique afin de ne pas oublier un symbole documenté ailleurs.

### Contrat comportemental

La table normative de la documentation publique précise, pour chaque opération :

- sa signature publique ;
- les types auxquels elle s’applique ;
- le résultat et les invariants promis ;
- les entrées vectorielles et matricielles disponibles ;
- les erreurs attendues ;
- les éventuelles restrictions mathématiques pour les modèles discrets, mixtes ou singuliers ;
- si l’opération est garantie par un fallback générique ou doit être implémentée par la famille.

L’inventaire de #426 doit couvrir :

- les noms exportés ou déclarés publics par `Copulas.jl` ;
- les fonctions actuellement non publiques mais documentées comme telles, notamment les mesures de dépendance et `measure`, afin de résoudre leur statut ;
- les extensions de `Distributions.jl`, `StatsBase.jl` et des autres interfaces adoptées ;
- les constructeurs et leurs garanties de validation et de stabilité de type ;
- `SklarDist`, `CopulaModel`, les générateurs exportés et les représentations spectrales publiques.

Chaque opération doit ensuite être classée conformément à #428 :

1. contrat universel de toute copule ;
2. contrat public dont la sémantique dépend de la nature mathématique de la copule ;
3. API autonome qui ne constitue pas une propriété de chaque copule ;
4. mécanisme strictement interne.

Cette classification doit être terminée avant de figer les helpers. Les helpers encodent une API décidée ; ils ne doivent pas la définir implicitement.

## 2. Contrat public exécuté copule par copule

Créer des helpers courts par groupe cohérent d’opérations :

```julia
test_constructors(C)
test_distribution_contract(C)
test_density_contract(C)
test_subsetting_contract(C)
test_conditioning_contract(C)
test_rosenblatt_contract(C)
test_dependence_contract(C)
test_fitting_contract(CT, data; method)
```

Une fonction de haut niveau applique l’ensemble du contrat à chaque entrée du bestiaire :

```julia
test_copula_contract(C; fitting_cases=...)
```

Elle appelle tous les groupes pertinents selon les règles définies dans la table normative. Les particularités ne sont pas déterminées par des prédicats propres à chaque instance, mais par quelques catégories mathématiques publiques et stables.

### Constructeurs

Vérifier pour chaque famille :

- les interfaces `MyCopula{d}(...)` et `MyCopula(d, ...)` promises ;
- l’égalité des modèles construits par les chemins équivalents ;
- la reconstruction par `params` ;
- la validation des dimensions et paramètres ;
- la stabilité de type du chemin paramétré par la dimension ;
- les réductions vers des copules limites lorsqu’elles font partie du contrat.

### Interface `Distributions.jl`

Vérifier pour chaque copule :

- `length`, `eltype`, `params`, support et frontières ;
- `cdf` et `logcdf` ;
- `rand` pour une observation et plusieurs observations, avec formes et types corrects ;
- marges uniformes ;
- `pdf`, `logpdf` et `loglikelihood` lorsque la notion de densité ordinaire s’applique ;
- comportement public documenté dans le cas contraire.

L’échantillonnage et la densité peuvent rester dans des helpers séparés pour la lisibilité, mais appartiennent au même contrat de distribution.

### Sous-ensembles

Vérifier :

- `subsetdims` ;
- conservation et ordre des dimensions demandées ;
- composition des sous-ensembles ;
- validations d’indices ;
- cohérence via `SklarDist`.

### Conditionnement

Vérifier :

- conditionnement scalaire et multiple ;
- chemins `Copula` et `SklarDist` ;
- support, frontières, monotonie de la CDF et quantile généralisé ;
- distributions conditionnelles continues, discrètes et mixtes ;
- validations d’indices et de valeurs.

### Rosenblatt

Vérifier :

- transformations directe et inverse ;
- entrées vectorielles et matricielles ;
- formes et types de sortie ;
- aller-retour lorsque la bijection est mathématiquement garantie ;
- sémantique publique prévue pour les copules singulières ou mixtes.

### Mesures de dépendance

Vérifier toutes les mesures retenues dans l’API normative :

- mesures scalaires comme `τ`, `ρ`, `β`, `γ` et `ι` ;
- dépendances de queue inférieure et supérieure ;
- variantes pairwise ;
- symétrie, diagonale et bornes attendues.

Le fait qu’une famille utilise une forme fermée ou un fallback ne change pas le contrat. Un test de chemin séparé garantit que chaque mécanisme interne est exercé.

### Ajustement

Vérifier pour chaque famille :

- `fit` via toutes les méthodes annoncées par la famille ;
- type et validité du modèle obtenu ;
- cohérence minimale du résultat ;
- erreurs sur des données ou méthodes incompatibles.

Les méthodes d’ajustement réellement disponibles peuvent varier par famille, mais cette variation doit être déclarée par l’interface d’ajustement elle-même et non reconstruite dans les tests.

## 3. Contrats publics complémentaires

Certaines interfaces ne se testent pas copule par copule, ou possèdent leur propre objet principal :

```julia
test_sklar_contract(D)
test_model_result_contract(M::CopulaModel)
test_pseudos()
test_measure()
test_nataf()
test_generator_public_api()
test_discrete_spectral_public_api()
```

`test_sklar_contract` couvre construction, paramètres, `cdf`, densité, échantillonnage, sous-ensemble, conditionnement et transformations marginales sans recopier inutilement tout le contrat de la copule.

`test_model_result_contract` couvre notamment `nobs`, `coef`, `coefnames`, `vcov`, `stderror`, `confint`, AIC, BIC, déviance, résidus et prédiction lorsque ces opérations sont promises.

La liste exacte des utilitaires autonomes doit venir de l’inventaire de l’API.

## 4. Contrats des composants internes

Tester directement les composants partagés au lieu de réassembler chaque combinaison possible :

```text
components/
  generators.jl
  tails.jl
  distortions.jl
  radial_distributions.jl
  samplers.jl
```

Pour chaque générateur : fonction, inverse, dérivées, monotonie, frontières et formes fermées.

Pour chaque tail EV : homogénéité de `ℓ`, marges, Pickands lorsque disponible, dérivées partielles et représentation spectrale éventuelle.

Pour chaque distortion : CDF monotone, quantile généralisé, support, atomes éventuels et référence analytique.

Ces tests internes ne remplacent jamais le contrat public copule par copule. Ils localisent les erreurs et évitent seulement de répéter les validations mathématiques détaillées à travers toutes les compositions.

## 5. Tests par chemin de dispatch

Créer un registre central et explicite contenant un ou deux représentants par mécanisme :

```julia
const PATH_CASES = (
    generic_cdf            = SomeCopula(...),
    generic_density        = SomeCopula(...),
    matrix_sampler         = ClaytonCopula{5}(...),
    frailty_sampler        = FrankCopula{3}(...),
    biv_ev_distortion      = GalambosCopula{2}(...),
    generic_condition      = RafteryCopula{2}(...),
    singular_condition     = MCopula{2}(),
    numerical_ev           = HuslerReissCopula{3}(...),
    fractional_williamson  = LiouvilleCopula{2}(...),
)
```

Ce registre vérifie les fast paths et fallbacks sans soumettre chaque famille aux mêmes comparaisons coûteuses.

Il doit remplacer les prédicats historiques comme `can_pdf`, `can_ad`, `check_rosenblatt`, `check_corkendall`, `can_integrate_pdf` ou `check_biv_conditioning`. Ces prédicats reconstituent actuellement une API parallèle dans les tests et deviennent rapidement faux.

## 6. Régressions propres aux familles

Les fichiers familiaux conservent uniquement ce qui distingue réellement la famille :

- valeurs de référence publiées ;
- formes fermées ;
- limites particulières ;
- bugs numériques déjà rencontrés ;
- algorithmes spécifiques.

Les erreurs de constructeurs relèvent du contrat commun, même si leurs paramètres particuliers sont fournis par les fixtures familiales. Aucun test générique de forme, support ou conditionnement ne doit être recopié ici.

## Structure de fichiers proposée

```text
test/
  runtests.jl
  fixtures.jl

  contracts/
    copulas.jl
    constructors.jl
    distribution.jl
    density.jl
    subsetting.jl
    conditioning.jl
    rosenblatt.jl
    dependence.jl
    fitting.jl
    sklar.jl
    model_results.jl
    utilities.jl

  components/
    generators.jl
    tails.jl
    distortions.jl
    radial_distributions.jl
    samplers.jl

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

## Couverture sans produit cartésien

Le contrat public est testé pour chaque copule, mais les validations coûteuses des algorithmes sous-jacents ne doivent pas être répétées pour chaque combinaison.

Inventorier les axes indépendants :

- dimensions 2, 3 et supérieure ;
- `Float32`, `Float64`, `BigFloat` lorsque promis ;
- modèle régulier, singulier et mixte ;
- paramètres intérieurs et cas limites ;
- formule fermée et fallback numérique ;
- frailty continue, discrète ou Williamson générique ;
- ordre entier et fractionnaire ;
- sampler direct, frailty, spectral ou générique ;
- conditionnement spécialisé et fallback ;
- CDF analytique, quadrature et noyau probabiliste.

Chaque copule reçoit un test minimal de chaque opération publique. Un petit ensemble transversal reçoit les validations numériques ou statistiques approfondies afin de couvrir tous les axes et chemins sans tester leur produit cartésien.

La matrice de couverture doit être une donnée Julia lisible, pas un document séparé susceptible de devenir obsolète.

## Réduction des tests coûteux

Retirer du bestiaire global :

- l’intégration de chaque densité ;
- les comparaisons échantillonnage/CDF répétées ;
- `corkendall` sur chaque modèle ;
- les validations statistiques de Rosenblatt sur toutes les variantes ;
- les comparaisons systématiques fast path/fallback ;
- les boucles sur de nombreux points qui compilent le même chemin.

Conserver dans le contrat de chaque copule un appel minimal à chaque opération publique. Reporter les contrôles approfondis vers les composants et chemins représentatifs :

- une intégration de densité par mécanisme ;
- une validation statistique par sampler ;
- un aller-retour approfondi par mécanisme de conditionnement ;
- des tests analytiques ciblés pour les familles singulières ;
- quelques points pour les frontières et l’intérieur.

Réduire `rand(C, 1000)` à `rand(C, 10)` change peu lorsque la compilation domine. Éviter une combinaison composite redondante supprime en revanche entièrement sa compilation.

## Stratégie CI

Conserver un seul processus par version Julia afin de partager la compilation entre fichiers.

- Julia 1.11, nouvelle version minimale : suite complète et couverture tant que la LTS officielle est plus ancienne.
- Julia stable : au minimum le contrat public complet, les constructeurs et les principaux chemins de dispatch.
- Tests numériques lourds : version minimale uniquement si nécessaire.
- Benchmarks : workflow séparé.
- Extensions : jobs séparés uniquement si elles nécessitent des environnements distincts.

Lorsque la LTS officielle devient compatible avec le minimum du paquet, le sélecteur `lts` peut remplacer le numéro explicite. Éclater chaque famille dans un job parallèle recompilierait Copulas.jl et ses dépendances dans chaque job ; le gain mural doit être mesuré avant d’adopter cette stratégie.

## Ordre d’implémentation

1. Inventorier séparément les symboles `export`/`public`, les symboles documentés et les méthodes publiques étendant des dépendances.
2. Relever la compatibilité minimale à Julia 1.11 dans `Project.toml`, adapter tous les workflows et environnements, et enregistrer cette rupture dans la version et les notes de release.
3. Décider explicitement quels symboles supplémentaires doivent devenir `public`, exportés ou internes, puis déclarer directement cette visibilité dans le module.
4. Écrire la table normative de l’API publique, puis aligner la page API, le manuel et le guide développeur conformément à #426 et #428.
5. Ajouter un chronométrage par fichier et établir la baseline de #425.
6. Créer `fixtures.jl`, le bestiaire et les helpers de contrats sans supprimer de tests.
7. Faire passer chaque copule par le contrat public complet.
8. Construire la matrice des chemins de dispatch demandée par #424.
9. Migrer puis supprimer `GenericTests.jl`.
10. Supprimer les duplications des fichiers familiaux.
11. Réorganiser les fichiers seulement après stabilisation du contenu.
12. Comparer temps total, temps de compilation et nombre de `MethodInstance`.

Le premier objectif structurel est de remplacer entièrement `GenericTests.jl` par un contrat public explicite, appliqué à chaque copule, et par un registre séparé des chemins internes. L’inventaire documentaire préalable évite de transformer les hypothèses historiques des tests en nouvelle API par accident.
