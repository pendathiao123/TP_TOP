import matplotlib.pyplot as plt

# Données mises à jour
matrix_sizes = [256, 512, 1024, 2048]  # Tailles des matrices (M = N = K)
execution_times = [0.127798, 1.332727, 21.714101, 337.949603]  # Temps d'exécution en secondes
flops = [0.26, 0.20, 0.10, 0.05]  # Performance en GFLOP/s

# Création des sous-graphiques
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 lignes, 1 colonne

# Tracé des performances (GFLOP/s)
axs[0].plot(matrix_sizes, flops, marker='o', label="Performance (GFLOP/s)")
axs[0].set_title("Performance du produit matriciel naïf (puissances de 2)")
axs[0].set_xlabel("Taille de la matrice (M = N = K)")
axs[0].set_ylabel("Performance (GFLOP/s)")
axs[0].grid(True)
axs[0].legend()

# Tracé des temps d'exécution
axs[1].plot(matrix_sizes, execution_times, marker='o', color='red', label="Temps d'exécution (s)")
axs[1].set_title("Temps d'exécution du produit matriciel naïf (puissances de 2)")
axs[1].set_xlabel("Taille de la matrice (M = N = K)")
axs[1].set_ylabel("Temps d'exécution (s)")
axs[1].grid(True)
axs[1].legend()

# Ajustement des espaces entre les sous-graphiques
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Ajoute un espacement vertical entre les graphiques

# Enregistrer l'image combinée
plt.savefig("naive_matrix_product_combined.png")

# Afficher les graphiques
plt.show()