import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PlotDrawer:
    def draw(self, data, best_medoids, clusters):
        data = np.array(data)
        best_medoids = np.array(best_medoids)
        clusters = np.array(clusters)

        unique_clusters = set(clusters)
        palette = sns.color_palette("hsv", len(unique_clusters))
        cluster_colors = {
            cluster: palette[i] for i, cluster in enumerate(unique_clusters)
        }

        for i, cluster in enumerate(unique_clusters):
            cluster_points = data[clusters == cluster]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=cluster_colors[cluster],
                label=f"Cluster: {i}",
            )  # Punkty danych

        plt.scatter(
            best_medoids[:, 0],
            best_medoids[:, 1],
            color="red",
            marker="X",
            s=100,
            label="Medoidy",
        )  # Medoidy

        # Opcjonalnie, dodanie etykiet i tytułu
        plt.title("Wykres punktów danych oraz medoidów")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Dodanie legendy
        plt.legend()

        # Wyświetlenie wykresu
        plt.grid(True)
        plt.show()
