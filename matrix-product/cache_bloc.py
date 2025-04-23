import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats depuis le fichier CSV
data = pd.read_csv("results_threads_blocks.csv")

# Obtenir les tailles de blocs uniques
block_sizes = data["BlockSize"].unique()

# Tracer les performances pour chaque taille de bloc
plt.figure(figsize=(12, 6))
for block_size in block_sizes:
    subset = data[data["BlockSize"] == block_size]
    plt.plot(subset["Threads"], subset["Performance"], marker="o", label=f"Block Size {block_size}")

# Ajouter les titres et légendes
plt.title("Performance vs Number of Threads for Different Block Sizes")
plt.xlabel("Number of Threads")
plt.ylabel("Performance (GFLOP/s)")
plt.grid(True)
plt.legend(title="Block Size")

# Forcer les ticks de l'axe des abscisses à 1, 2, 3, 4
plt.xticks([1, 2, 3, 4])

# Sauvegarder et afficher le graphique
plt.savefig("performance_vs_threads_blocks.png")
plt.show()