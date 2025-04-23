import matplotlib.pyplot as plt

# Données pour LayoutRight
layout_right = {
    "Threads": [1, 2, 3, 4],  # Nombre de threads
    "ExecutionTime": [45.022297, 21.756029, 31.281163, 25.012962],  # Temps d'exécution en secondes
    "GFLOP/s": [0.36, 0.74, 0.51, 0.64]  # Performances en GFLOP/s
}

# Données pour LayoutLeft
layout_left = {
    "Threads": [1, 2, 3, 4],  # Nombre de threads
    "ExecutionTime": [44.334248, 20.377061, 29.438634, 22.543491],  # Temps d'exécution en secondes
    "GFLOP/s": [0.36, 0.79, 0.54, 0.71]  # Performances en GFLOP/s
}

# Création des sous-graphiques
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 lignes, 1 colonne

# Tracé des performances (GFLOP/s)
axs[0].plot(layout_right["Threads"], layout_right["GFLOP/s"], marker='o', label="LayoutRight")
axs[0].plot(layout_left["Threads"], layout_left["GFLOP/s"], marker='o', label="LayoutLeft")
axs[0].set_title("Comparaison des performances : LayoutRight vs LayoutLeft")
axs[0].set_xlabel("Nombre de threads")
axs[0].set_ylabel("Performance (GFLOP/s)")
axs[0].set_xticks([1, 2, 3, 4])  # Forcer les ticks sur l'axe x
axs[0].grid(True)
axs[0].legend()

# Tracé des temps d'exécution
axs[1].plot(layout_right["Threads"], layout_right["ExecutionTime"], marker='o', label="LayoutRight")
axs[1].plot(layout_left["Threads"], layout_left["ExecutionTime"], marker='o', label="LayoutLeft")
axs[1].set_title("Comparaison des temps d'exécution : LayoutRight vs LayoutLeft")
axs[1].set_xlabel("Nombre de threads")
axs[1].set_ylabel("Temps d'exécution (s)")
axs[1].set_xticks([1, 2, 3, 4])  # Forcer les ticks sur l'axe x
axs[1].grid(True)
axs[1].legend()

# Ajustement des espaces entre les sous-graphiques
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Ajoute un espacement vertical entre les graphiques

# Enregistrer l'image combinée
plt.savefig("layout_comparison_combined.png")

# Afficher les graphiques
plt.show()