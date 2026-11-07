"""
Dynamique moléculaire 2D d'un système de Lennard-Jones chauffé par paliers.

N particules dans une boîte carrée périodique (convention de l'image minimum)
interagissent par le potentiel de Lennard-Jones. Les équations de Newton sont
intégrées par l'algorithme de Verlet vitesse. La température est augmentée
par paliers (ré-échelonnage des vitesses), puis le système évolue librement
pendant 300 pas à chaque palier. On mesure l'énergie cinétique, potentielle et
totale, la courbe calorique <E>(T) et la capacité calorifique Cv = d<E>/dT.

Unités réduites : sigma = epsilon = m = k_B = 1.

Projet M1 Physique, CY Cergy Paris Université (février 2026).

Utilisation :
    python dynamique_moleculaire.py            # calcul, figures et animation
    python dynamique_moleculaire.py --sauver   # enregistre figures et GIF dans figures/
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SAUVER = "--sauver" in sys.argv


# --- 1. MOTEUR PHYSIQUE ---
class SimulationMD_2D:
    def __init__(self, N, L_boite, dt):
        self.N, self.L_boite, self.dt = N, L_boite, dt
        self.positions = np.zeros((N, 2))
        self.vitesses = np.zeros((N, 2))
        self.forces = np.zeros((N, 2))

    def initialiser_positions_carre(self):
        """Réseau carré légèrement perturbé."""
        n_cote = int(np.ceil(self.N**(1/2)))
        espacement = self.L_boite / n_cote
        k = 0
        for x in range(n_cote):
            for y in range(n_cote):
                if k < self.N:
                    self.positions[k] = [x * espacement + espacement/2, y * espacement + espacement/2]
                    self.positions[k] += np.random.uniform(-0.02, 0.02, 2)
                    k += 1

    def initialiser_vitesses(self, T_cible):
        """Thermostat par ré-échelonnage : impose T = Ec / N (2 degrés de liberté par particule)."""
        if np.sum(self.vitesses**2) == 0:
            self.vitesses = np.random.randn(self.N, 2)
        self.vitesses -= np.mean(self.vitesses, axis=0)  # Stop dérive du centre de masse
        Ec = 0.5 * np.sum(self.vitesses**2)
        T_inst = Ec / self.N
        if T_inst > 0:
            self.vitesses *= np.sqrt(T_cible / T_inst)

    def calculer_physique(self):
        """Forces et énergie potentielle de Lennard-Jones, vectorisées (O(N²))."""
        vect = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        vect -= self.L_boite * np.round(vect / self.L_boite)  # Image minimum
        r2 = np.sum(vect**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        inv_r6 = (1.0 / r2)**3
        inv_r12 = inv_r6**2
        # Force de Lennard-Jones : F = 24 (2/r^12 - 1/r^6) r_vec / r²
        facteur = 24.0 * (1.0/r2) * (2.0 * inv_r12 - inv_r6)
        F = np.sum(facteur[:, :, np.newaxis] * vect, axis=1)
        Ep = 0.5 * np.sum(4.0 * (inv_r12 - inv_r6))
        return F, Ep

    def step(self):
        """Un pas de Verlet vitesse ; renvoie (Ec, Ep)."""
        self.positions += self.vitesses * self.dt + 0.5 * self.forces * self.dt**2
        self.positions %= self.L_boite
        F_new, Ep = self.calculer_physique()
        self.vitesses += 0.5 * (self.forces + F_new) * self.dt
        self.forces = F_new
        return 0.5 * np.sum(self.vitesses**2), Ep


# --- 2. CONFIGURATION ---
np.random.seed(1)  # Graine fixe : résultats reproductibles
N = 256
L = 40.0
dt = 0.005
N_PALIERS = 40        # Paliers de température : T* = 0.1, 0.13, ..., 1.27
PAS_PAR_PALIER = 300  # Pas de thermalisation à chaque palier

sim = SimulationMD_2D(N, L, dt)
sim.initialiser_positions_carre()
sim.forces, _ = sim.calculer_physique()

all_pos = []
temps_data, Ec_data, Ep_data, Etot_data = [], [], [], []
T_list, E_moy_list = [], []

print("Calcul de la simulation...")

# --- 3. BOUCLE DE CALCUL : CHAUFFAGE PAR PALIERS ---
for t_idx in range(N_PALIERS):
    T_cible = 0.1 + t_idx * 0.03
    sim.initialiser_vitesses(T_cible)

    e_paliers = []
    for _ in range(PAS_PAR_PALIER):
        Ec, Ep = sim.step()
        all_pos.append(sim.positions.copy())
        temps_data.append(len(temps_data) * dt)
        Ec_data.append(Ec); Ep_data.append(Ep); Etot_data.append(Ec + Ep)
        e_paliers.append(Ec + Ep)

    T_list.append(T_cible)
    E_moy_list.append(np.mean(e_paliers))

print("Calcul terminé.")

# Contrôle de l'intégrateur : dérive de l'énergie totale à l'intérieur de chaque palier
E_paliers = np.array(Etot_data).reshape(N_PALIERS, PAS_PAR_PALIER)
derive = np.max(np.abs(E_paliers[:, -1] - E_paliers[:, 0]) / np.abs(E_paliers[:, 0]))
print(f"Dérive relative maximale de l'énergie totale sur un palier : {derive * 100:.2f} %")

# --- 4. FIGURES ---

# Figure 1 : énergies cinétique, potentielle et totale
fig1 = plt.figure("Analyse des Energies", figsize=(10, 5))
plt.plot(temps_data, Ec_data, label='K (Cinétique)', color='tab:blue', alpha=0.7)
plt.plot(temps_data, Ep_data, label='U (Potentielle)', color='tab:orange', alpha=0.7)
plt.plot(temps_data, Etot_data, 'k-', label='E totale', linewidth=2)
plt.title("Évolution des énergies lors du chauffage par paliers")
plt.xlabel("Temps réduit"); plt.ylabel("Énergie"); plt.grid(); plt.legend()
plt.tight_layout()

# Figure 2 : courbe calorique et Cv calculé sur les moyennes de palier
fig2 = plt.figure("Courbe calorique", figsize=(10, 6))
T_arr = np.array(T_list)
E_arr = np.array(E_moy_list)
Cv = np.diff(E_arr) / np.diff(T_arr)
T_mid = (T_arr[:-1] + T_arr[1:]) / 2

plt.subplot(211)
plt.plot(T_arr, E_arr, 'go-')
plt.ylabel("<E> (moyenne par palier)"); plt.title("Courbe calorique"); plt.grid()
plt.subplot(212)
plt.plot(T_mid, Cv, 'r-s')
plt.axhline(0, color='gray', lw=0.8)
plt.ylabel("Cv = d<E>/dT"); plt.xlabel("T*"); plt.grid()
plt.title("Capacité calorifique (différences finies entre paliers)")
plt.tight_layout()

# Figure 3 : animation des particules, trace des 3 premières
def creer_animation(tous_les=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    pts, = ax.plot([], [], 'bo', ms=5 if SAUVER else 8)
    trajs_plots = [ax.plot([], [], '-', lw=1.5, alpha=0.6)[0] for _ in range(3)]
    tx, ty = [[] for _ in range(3)], [[] for _ in range(3)]
    titre = ax.set_title("")

    def animate(k):
        i = k * tous_les
        pos = all_pos[i]
        pts.set_data(pos[:, 0], pos[:, 1])
        for j in range(3):  # Trace "comète" des 3 premières particules
            tx[j].append(pos[j, 0]); ty[j].append(pos[j, 1])
            longueur = max(50 // tous_les, 6)
            trajs_plots[j].set_data(tx[j][-longueur:], ty[j][-longueur:])
        titre.set_text(f"T* = {T_list[i // PAS_PAR_PALIER]:.2f}")
        return [pts, titre] + trajs_plots

    n_images = len(all_pos) // tous_les
    return fig, animation.FuncAnimation(fig, animate, frames=n_images, interval=10, blit=not SAUVER)


if SAUVER:
    os.makedirs("figures", exist_ok=True)
    fig1.savefig("figures/energies.png", dpi=90)
    fig2.savefig("figures/courbe_calorique.png", dpi=90)
    fig3, ani = creer_animation(tous_les=40)  # une image tous les 40 pas
    ani.save("figures/animation.gif", writer=animation.PillowWriter(fps=15), dpi=60)
    plt.close("all")
    print("Figures enregistrées dans figures/")
else:
    fig3, ani = creer_animation()
    plt.show()
