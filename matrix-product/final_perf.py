import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats depuis le fichier CSV
data = pd.read_csv("results_threads_blocks.csv")

# Tracer les performances pour chaque taille de bloc
block_sizes = data["BlockSize"].unique()

plt.figure(figsize=(10, 6))
for block_size in block_sizes:
    subset = data[data["BlockSize"] == block_size]
    plt.plot(subset["Threads"], subset["Performance"], marker='o', label=f"Block Size {block_size}")

# Configurer le graphique
plt.title("Performance Scaling with Threads and Block Sizes")
plt.xlabel("Number of Threads")
plt.ylabel("Performance (GFLOP/s)")
plt.legend()
plt.grid(True)

# Sauvegarder et afficher le graphique
plt.savefig("final_performance_scaling.png")
plt.show()