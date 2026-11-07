# Dynamique moléculaire 2D : un système de Lennard-Jones chauffé par paliers

Simulation de **dynamique moléculaire** de 256 particules interagissant par le **potentiel de Lennard-Jones**, dans une boîte carrée périodique. Le système est chauffé progressivement, de $T^* = 0{,}1$ à $T^* = 1{,}27$, et on suit ses énergies, sa courbe calorique $\langle E \rangle(T)$ et sa capacité calorifique $C_v = d\langle E \rangle / dT$.

![Animation des 256 particules pendant le chauffage](figures/animation.gif)

*Les 256 particules pendant les 40 paliers de chauffage (une image tous les 40 pas). Les particules partent d'un réseau carré, se regroupent en agrégats à basse température, puis ces agrégats se dispersent quand la température augmente.*

> **En bref**
> - Intégrateur de **Verlet vitesse**, conditions aux limites périodiques avec **image minimum**, forces calculées de façon vectorisée avec NumPy.
> - Chauffage par paliers : thermostat par ré-échelonnage des vitesses, puis 300 pas d'évolution libre à chaque température.
> - Sur chaque palier, l'énergie totale est conservée à mieux que 0,2 %, ce qui valide l'intégrateur.
> - À cette faible densité ($\rho^* = 0{,}16$), on observe surtout une **condensation en gouttelettes** puis leur **évaporation**, plutôt qu'une fusion de cristal (voir § 5).

## 1. Objectif

Simuler un système atomique simple à partir des seules équations de Newton, et observer comment l'ordre et l'énergie du système évoluent avec la température.

## 2. Le modèle

### Potentiel de Lennard-Jones

$$V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

Le terme en $r^{-12}$ modélise la répulsion à courte distance (exclusion de Pauli), le terme en $r^{-6}$ l'attraction de Van der Waals. Le minimum du potentiel est atteint en $r = 2^{1/6}\sigma \approx 1{,}12\thinspace\sigma$. Les forces dérivent du potentiel :

$$\vec F_{ij} = \frac{24\varepsilon}{r_{ij}^2}\left[2\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^{6}\right]\vec r_{ij}$$

On travaille en **unités réduites** : $\sigma = \varepsilon = m = k_B = 1$.

### Conditions aux limites périodiques

Une particule qui sort d'un côté de la boîte rentre par le côté opposé, ce qui simule un milieu sans bords. Avec la **convention de l'image minimum**, chaque particule interagit avec l'image la plus proche de chacune des autres.

### Verlet vitesse

$$\vec r(t + \Delta t) = \vec r(t) + \vec v(t)\thinspace\Delta t + \tfrac{1}{2}\vec F(t)\thinspace\Delta t^2$$

$$\vec v(t + \Delta t) = \vec v(t) + \tfrac{1}{2}\left[\vec F(t) + \vec F(t + \Delta t)\right]\Delta t$$

Contrairement au schéma d'Euler, Verlet vitesse est **symplectique** : l'énergie totale ne dérive pas sur les temps longs, elle oscille autour de sa valeur exacte.

### Température et chauffage

En 2D, chaque particule a deux degrés de liberté, d'où $E_c = N k_B T$ et $T = E_c / N$. À chaque palier, les vitesses sont multipliées par $\sqrt{T_\text{cible} / T}$ (thermostat par ré-échelonnage), puis le système évolue librement pendant 300 pas : l'énergie se redistribue entre cinétique et potentielle par les collisions.

## 3. Implémentation

| Méthode | Rôle |
|---|---|
| `initialiser_positions_carre()` | réseau carré de $16 \times 16$ sites, légèrement perturbé |
| `initialiser_vitesses(T)` | tirage gaussien, suppression de la dérive du centre de masse, ré-échelonnage à $T$ |
| `calculer_physique()` | forces et énergie potentielle par diffusion de tableaux NumPy (toutes les paires, en $O(N^2)$) |
| `step()` | un pas de Verlet vitesse, renvoie $(E_c, E_p)$ |

| Paramètre | Valeur | Rôle |
|---|---|---|
| `N` | 256 | nombre de particules |
| `L` | 40 | côté de la boîte, soit une densité $\rho^* = N / L^2 = 0{,}16$ |
| `dt` | 0,005 | pas de temps |
| paliers | 40, de $T^* = 0{,}1$ à $1{,}27$ par pas de 0,03 | chauffage |
| pas par palier | 300 | thermalisation |

Le pas de temps a été choisi par essais : avec $\Delta t = 0{,}02$, deux particules peuvent se rapprocher assez pour que la force répulsive en $r^{-13}$ fasse « exploser » la simulation.

## 4. Résultats

![Énergies au cours du chauffage](figures/energies.png)

L'**énergie totale** (en noir) forme des **plateaux horizontaux** : entre deux ré-échelonnages, elle est conservée à mieux que 0,2 %. C'est la signature d'un intégrateur symplectique bien réglé. Les sauts entre plateaux correspondent à l'énergie injectée ou retirée par le thermostat.

![Courbe calorique et capacité calorifique](figures/courbe_calorique.png)

## 5. Interprétation

Le comportement observé dépend fortement de la **densité**. Avec 256 particules dans une boîte de côté 40, le réseau initial a un pas de 2,5 σ, plus de deux fois la distance d'équilibre du potentiel (1,12 σ). L'état initial n'est donc pas un cristal compact, mais un gaz dilué disposé sur une grille.

- **De $T^* = 0{,}1$ à $0{,}35$ environ : condensation.** Les particules s'attirent et forment des **agrégats** (les « gouttelettes » de l'animation). L'énergie potentielle chute de −10 à −200 environ, et l'énergie libérée réchauffe le système, que le thermostat refroidit ensuite. L'énergie moyenne **diminue** alors que la consigne de température augmente, d'où un $C_v$ apparent négatif : le système n'est pas à l'équilibre sur ces paliers, il est encore en train de s'agréger.
- **Au-delà : évaporation progressive.** L'énergie potentielle remonte, les agrégats se défont, et à $T^* = 1{,}27$ le système est un fluide dilué.
- **Capacité calorifique.** Calculée par différences finies entre paliers successifs, elle fluctue d'un facteur 2 à 5 d'un palier à l'autre. Le pic le plus haut se situe vers $T^* \approx 0{,}84$, mais d'autres pics de taille comparable apparaissent vers 1,08 et 1,14 : avec une seule simulation et 300 pas par palier, ce pic n'est pas statistiquement significatif.

Pour comparaison, les simulations publiées du système de Lennard-Jones 2D placent le point triple vers $T^* \approx 0{,}40$ et le point critique liquide-gaz vers $T^* \approx 0{,}46$ (potentiel tronqué). À $\rho^* = 0{,}16$, sous la densité critique (≈ 0,37), le système traverse donc une coexistence gaz-agrégats, puis devient un fluide homogène.

## 6. Limites et pistes

- **Partir d'un vrai cristal** : un réseau triangulaire avec un pas proche de 1,12 σ, soit une densité d'environ 0,8, permettrait d'observer la **fusion** proprement dite.
- **Mesurer $C_v$ par les fluctuations** : dans l'ensemble canonique, $C_v = (\langle E^2 \rangle - \langle E \rangle^2) / k_B T^2$. Combiné à des paliers plus longs, à des moyennes sur plusieurs simulations et à des barres d'erreur, cela donnerait une courbe bien moins bruitée.
- **Thermostat** : le ré-échelonnage brutal ne produit pas l'ensemble canonique ; un thermostat de Berendsen, d'Andersen ou de Nosé-Hoover serait plus rigoureux.
- **Observables structurales** : fonction de distribution radiale $g(r)$ et paramètre d'ordre orientationnel $\psi_6$ pour distinguer solide, liquide et gaz ; déplacement quadratique moyen pour le coefficient de diffusion ; pression par le théorème du viriel.
- **Performances** : le calcul des forces sur toutes les paires est en $O(N^2)$. Un rayon de coupure (2,5 σ) et des listes de cellules le ramènent à $O(N)$.

## 7. Lancer le code

```bash
pip install -r requirements.txt
python dynamique_moleculaire.py            # calcul (environ 1 min), figures et animation
python dynamique_moleculaire.py --sauver   # enregistre les figures et le GIF dans figures/
```

La graine aléatoire est fixée : chaque exécution redonne les mêmes résultats.

## 8. Références

- M. P. Allen et D. J. Tildesley, *Computer Simulation of Liquids*, Oxford University Press (2e éd., 2017).
- D. Frenkel et B. Smit, *Understanding Molecular Simulation*, Academic Press (2e éd., 2002).
- *Phase behaviour of Lennard-Jones particles in two dimensions*, arXiv:2010.03876 (2020).

---

Projet réalisé en M1 Physique à CY Cergy Paris Université (février 2026).
