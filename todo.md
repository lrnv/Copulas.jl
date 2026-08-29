# Réduction du temps de la suite de tests

Référence CI du 28 août 2026 : **13 min 30,8 s** pour 8 785 tests
(8 784 réussis, 1 broken). La cible indicative est **6 min 45 s**, mais elle
ne justifie jamais une perte de couverture ou une dégradation d'oracle.

## Règles non négociables

Toute optimisation doit conserver simultanément :

1. l'inventaire de chaque symbole, famille et comportement publics ;
2. l'exécution du noyau sémantique public pour chaque famille ;
3. l'exécution de chaque route distincte de dispatch, pour chaque classe de
   dimension pertinente ;
4. le registre des branches internes qui ne sont pas distinguées par `which` ;
5. un oracle indépendant ou une identité exacte pour chaque spécialisation ;
6. tous les cas mathématiquement distincts : intérieur, frontières,
   hors-support, coordonnées asymétriques et topologies de paramètres.

Le nombre brut d'assertions n'est pas une preuve. Réciproquement, une assertion
ne peut être supprimée que si une autre assertion identifiée prouve exactement
la même obligation sur la même route. Une comparaison entre deux appels au
même noyau ne constitue pas un oracle indépendant.

En particulier :

- ne pas réduire les matrices à une colonne : deux colonnes sont nécessaires
  pour détecter un adaptateur qui ne parcourt qu'une observation ;
- ne pas remplacer toutes les marges par une seule pour les modèles
  asymétriques ; couvrir chaque orbite de coordonnées distincte ;
- ne pas dédupliquer sur `which` seul lorsque le corps contient une branche
  selon la valeur, la représentation ou la dimension ;
- ne pas remplacer HCubature par une quadrature fixe sans conserver une
  référence indépendante pour chaque classe d'intégrande ;
- ne pas partager RNG, buffers, conditionnelles, caches mutables ou résultats
  de fitting entre tests. Seules les fixtures déterministes immuables peuvent
  être partagées.

## P0 — contrat universel

- [ ] Distinguer explicitement les assertions du noyau scalaire par famille et
  celles des adaptateurs de collections. Une exécution d'adaptateur peut être
  mutualisée seulement si sa clé inclut la méthode, la classe dimensionnelle
  et les branches comportementales pertinentes ; `applicable` doit rester
  vérifié pour chaque famille.
- [ ] Vérifier si les appels de CDF aux bornes et aux marges sélectionnent des
  branches internes distinctes. Ne mutualiser que ceux dont l'identité de
  chemin et d'obligation est démontrée.
- [ ] Dans le conditionnement, inventorier séparément les chemins scalaire,
  conjoint, continu et atomique avant toute réduction de points.

## P0 — dépendance et fitting

- [ ] Pour toute mesure stochastique, garder un appel de l'API publique avec son
  budget de production. Les propriétés statistiques peuvent utiliser un
  oracle déterministe moins coûteux séparé, mais jamais un chemin de production
  modifié uniquement pour les tests.
- [ ] Construire une clé de fitting composée de `_fit`, `_unbound_params`,
  `_rebound_params`, bornes, méthode, classe dimensionnelle et topologie des
  paramètres. L'optimiseur ne peut être mutualisé qu'entre clés identiques ;
  l'applicabilité et le round-trip restent vérifiés famille par famille.
- [ ] Réutiliser un résultat ajusté pour le contrat `CopulaModel` seulement si
  cela ne supprime pas l'appel public `fit(CopulaModel, ...)` lui-même.
- [ ] Garder au moins une Hessienne publique, les routes Sklar IFM et ECDF, et
  chaque estimateur EV (`ols`, `cfg`, `pickands`) en dimensions 2 et 3 lorsque
  l'algorithme diffère.

## P1 — régressions coûteuses

- [ ] Empirical EV : cartographier les routes des trois estimateurs en d=2/d=3.
  Réduire une grille seulement après comparaison à une référence indépendante
  conservée dans la suite.
- [ ] Extremal-t : remplacer les répétitions d'une même CDF numérique par des
  identités d'homogénéité/STDF, mais conserver une valeur numérique indépendante
  par implémentation distincte.
- [ ] Liouville : partager générateurs et lois radiales immuables ; conserver
  l'intégration simplex de référence et les identités radiale–Dirichlet pour
  toutes les classes (entière, fractionnaire, frailty et générique).
- [ ] Tables de régression : établir d'abord les classes d'équivalence des
  points ; ne retirer que les répétitions appartenant à la même classe et au
  même chemin.

## P1 — oracles numériques

- [ ] Identifier l'éventuel appel contractuel résiduel à la CDF Student et le
  remplacer par l'identité elliptique uniquement si celle-ci est indépendante
  de l'implémentation testée.
- [ ] Pour BigFloat, conserver au moins propagation de type, valeur numérique
  indépendante et une route réellement calculée par classe d'algorithme.
- [ ] Williamson : conserver CDF, PDF et quantile pour un ordre entier et un
  ordre réel, ainsi que toute représentation qui change le dispatch.
- [ ] Évaluer Gauss–Legendre seulement comme oracle supplémentaire. HCubature
  ne peut disparaître d'une classe d'intégrande qu'après validation analytique
  ou contre une constante haute précision enregistrée avec sa provenance.

## Mesure et CI

- [ ] Comparer trois runs du même runner et raisonner sur leur médiane.
- [ ] Ajouter d'abord des budgets en avertissement. Des seuils bloquants ne
  seront introduits qu'après mesure de la variance des runners, afin d'éviter
  une CI floconneuse.
- [ ] Comparer avant fusion les ensembles de familles, routes `which`, branches
  comportementales, classes dimensionnelles et entrées du proof ledger avec la
  référence. Aucun de ces ensembles ne peut diminuer.
- [ ] Si le temps séquentiel reste supérieur à huit minutes après les
  optimisations démontrées, répartir la CI en deux shards équilibrés. Le
  sharding réduit le temps mural, pas le budget CPU, et ne remplace aucun test.

Le fichier sera supprimé lorsque les optimisations démontrées auront été
appliquées et que la cible aura été observée sur trois runs consécutifs.
