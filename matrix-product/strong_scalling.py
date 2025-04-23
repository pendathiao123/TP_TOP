import matplotlib.pyplot as plt
import pandas as pd

# Données
data = {
    "Threads": [1, 2, 3, 4],
    "ExecutionTime": [45.022297, 21.756029, 31.281163, 25.012962],
    "GFLOP/s": [0.36, 0.74, 0.51, 0.64]
}

# Convertir les données en DataFrame
df = pd.DataFrame(data)

# Calculer l'accélération (speedup)
df["Speedup"] = df["ExecutionTime"].iloc[0] / df["ExecutionTime"]

# Création des sous-graphiques
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 lignes, 1 colonne

# Tracé des performances (GFLOP/s)
axs[0].plot(df["Threads"], df["GFLOP/s"], marker='o', label="Performance (GFLOP/s)")
axs[0].set_title("Performance en fonction du nombre de threads")
axs[0].set_xlabel("Nombre de threads")
axs[0].set_ylabel("Performance (GFLOP/s)")
axs[0].set_xticks([1, 2, 3, 4])  # Forcer les ticks sur l'axe x
axs[0].grid(True)
axs[0].legend()

# Tracé de l'accélération (speedup)
axs[1].plot(df["Threads"], df["Speedup"], marker='o', color='green', label="Accélération (Speedup)")
axs[1].set_title("Étude de scalabilité forte")
axs[1].set_xlabel("Nombre de threads")
axs[1].set_ylabel("Accélération (Speedup)")
axs[1].set_xticks([1, 2, 3, 4])  # Forcer les ticks sur l'axe x
axs[1].grid(True)
axs[1].legend()

# Ajustement des espaces entre les sous-graphiques
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Ajoute un espacement vertical entre les graphiques

# Enregistrer l'image combinée
plt.savefig("scaling_combined.png")

# Afficher les graphiques
plt.show()