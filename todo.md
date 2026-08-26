# Redesign de l’architecture des tests

## État de l’implémentation

- [x] Julia 1.11 est la version minimale et l’API SemVer est déclarée.
- [x] La table comportementale publique est documentée.
- [x] Le bestiaire compact et les registres indépendants de constructeurs,
  fitting et chemins de dispatch existent dans `test/fixtures.jl`.
- [x] Les contrats copule couvrent distribution, densité selon la nature
  mathématique, sous-ensembles, conditionnement, Rosenblatt et dépendance.
- [x] Les API autonomes `SklarDist`, `CopulaModel`, `pseudos`, `measure` et
  `Nataf` ont leurs propres contrats.
- [x] Les primitives publiques des générateurs et tails ont des contrats de
  composants, et les chemins internes coûteux ont un registre transversal.
- [x] Le bestiaire cartésien et les prédicats de capacité de
  `old/GenericTests.jl` ont été supprimés.
- [ ] Faire passer la nouvelle suite en CI, corriger les divergences révélées,
  puis migrer fichier par fichier les régressions historiques restantes.
- [ ] Enregistrer les temps par groupe et supprimer `test/old/` quand sa
  dernière régression utile a été reclassée.

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

La source de vérité doit donc avoir trois couches cohérentes :

1. `export` et `public` déclarent exhaustivement les symboles appartenant à l’API de Copulas.jl ;
2. une table normative dans la documentation publique décrit les comportements promis ;
3. les contrats de test vérifient ces comportements.

Le guide développeur explique seulement les points d’extension internes permettant de satisfaire ce contrat.

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
- les méthodes publiques de Copulas.jl, notamment les mesures de dépendance et `measure` ;
- les extensions de `Distributions.jl`, `StatsBase.jl` et des autres interfaces adoptées, notamment `fit`, `cdf`, `pdf`, `logpdf`, `loglikelihood`, `rand`, `corkendall` et `corspearman` ;
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
test_constructors(case)
test_distribution_contract(case, ctx)
test_density_contract(case, ctx)
test_subsetting_contract(case, ctx)
test_conditioning_contract(case, ctx)
test_rosenblatt_contract(case, ctx)
test_dependence_contract(case, ctx)
test_fitting_contract(case, ctx)
```

Une fonction de haut niveau applique l’ensemble du contrat à chaque entrée du bestiaire :

```julia
test_copula_contract(case)
```

Elle construit la copule une seule fois, prépare un petit contexte partagé (`u`, `U`, indices et probabilités intérieures), puis appelle tous les groupes. Aucun helper ne doit rééchantillonner ou reconstruire le même modèle sans nécessité.

Le bestiaire doit rester une donnée Julia simple, composée de tuples nommés et de modèles construits exclusivement avec l’API publique. Ne pas créer une hiérarchie de types ou une macro de fixtures. Séparer seulement les cohortes correspondant à une différence mathématique du contrat : copules absolument continues, singulières et mixtes. Les valeurs par défaut portent le contrat complet ; les cohortes non régulières ne changent que la sémantique de la densité et de l’inversion de Rosenblatt.

Maintenir trois registres indépendants lorsque leurs axes ne coïncident pas :

- `COPULA_CASES` pour le contrat commun sur des instances ;
- `CONSTRUCTOR_CASES` pour comparer les formes typées, dynamiques et éventuellement inférables ;
- `FITTING_CASES` pour les méthodes publiquement promises par chaque famille.

Ne pas interroger `_available_fitting_methods` dans les contrats : cette fonction est interne et non-SemVer. Les méthodes publiques d’ajustement doivent être déclarées explicitement par les fixtures à partir de la documentation normative.

Le bestiaire doit contenir à la fois les alias familiaux usuels et quelques compositions génériques réellement constructibles par l’API publique : générateur + `ArchimedeanCopula`, tail + `ExtremeValueCopula`, générateur + tail + `ArchimaxCopula`, générateur + paramètres de Dirichlet + `LiouvilleCopula`, transformations et `SklarDist`.

Les tests contractuels utilisent très peu d’observations et de points. Leur rôle est de vérifier que chaque opération existe et respecte ses invariants. Les validations statistiques, intégrations et comparaisons de formules appartiennent aux tests de chemins, composants ou régressions.

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
  Aqua.jl
  fixtures.jl

  old/                 # suite historique, toujours exécutée pendant la migration

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

La suite historique a été déplacée sans modification dans `test/old/` et reste
incluse par `runtests.jl`. Chaque migration vers les nouveaux contrats doit retirer
dans le même commit les assertions devenues redondantes du fichier historique
concerné. Le dossier `old/` disparaît lorsque sa dernière garantie utile a été
reclassée comme contrat public, test de composant, test de chemin ou régression
familiale.

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

Chaque point ci-dessous correspond autant que possible à un commit autonome. Le commit retire cette ligne du TODO et supprime simultanément les assertions historiques qu’il remplace.

1. Enregistrer dans #425 la baseline de la suite historique encore intacte, avec temps par fichier et temps total.
2. Écrire la table normative de l’API publique et l’utiliser pour figer les cohortes et registres de `fixtures.jl`.
3. Ajouter le driver `test_copula_contract`, le contexte partagé et le contrat des constructeurs ; migrer la partie correspondante de `old/Constructors.jl` et `old/GenericTests.jl`.
4. Ajouter le contrat `Distributions.jl` fondamental : dimension, type, paramètres, support, `cdf`, `logcdf` et échantillonnage vectoriel/matriciel ; retirer les doublons historiques.
5. Ajouter le contrat de densité et vraisemblance avec sa sémantique continue/singulière/mixte ; conserver les intégrations approfondies uniquement dans les tests de chemins.
6. Ajouter le contrat de `subsetdims` pour `Copula` et `SklarDist`, puis réduire `old/Subsetting.jl` aux seules régressions non génériques.
7. Ajouter le contrat de `condition` scalaire et multiple sur les échelles copule et Sklar ; migrer les invariants génériques de `old/ConditionalDistribution.jl`.
8. Ajouter le contrat des transformations de Rosenblatt vectorielles et matricielles, avec bijection seulement lorsqu’elle est promise mathématiquement.
9. Ajouter le contrat des mesures scalaires et pairwise, y compris les méthodes `StatsBase`, sans répéter une validation statistique coûteuse pour chaque modèle.
10. Ajouter les contrats de `fit` et `CopulaModel` à partir de `FITTING_CASES`, puis conserver dans `old/FittingTest.jl` seulement les régressions algorithmiques.
11. Ajouter le contrat complet de `SklarDist` sans recopier les validations déjà garanties par la copule sous-jacente.
12. Ajouter les contrats autonomes de `pseudos`, `measure`, `Nataf`, des générateurs publics et de la représentation spectrale publique.
13. Supprimer `old/GenericTests.jl` dès que toutes ses assertions utiles sont classées dans les contrats précédents, un test de chemin ou une régression familiale.
14. Construire la matrice des chemins de dispatch demandée par #424 et y déplacer les validations coûteuses représentatives.
15. Migrer les contrats des composants partagés : générateurs, tails, distortions, distributions radiales et samplers.
16. Répartir les dernières régressions utiles dans `families/` et `extensions/`, puis supprimer chaque fichier restant de `old/`.
17. Ajouter un chronométrage par groupe, comparer à la baseline de #425 et supprimer entièrement `test/old/`.

Le premier objectif structurel est de remplacer entièrement `GenericTests.jl` par un contrat public explicite, appliqué à chaque copule, et par un registre séparé des chemins internes. L’inventaire documentaire préalable évite de transformer les hypothèses historiques des tests en nouvelle API par accident.
