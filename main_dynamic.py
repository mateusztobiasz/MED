import random
import numpy as np

import matplotlib.pyplot as plt

def clarans(data, num_clusters, numlocal, maxneighbor):
    """
    CLARANS clustering algorithm operating on an abstract graph.

    Parameters:
    data (list of tuple of floats): Dataset to cluster.
    num_clusters (int): Number of clusters.
    numlocal (int): Number of local minima attempts.
    maxneighbor (int): Maximum number of neighbors to explore.

    Returns:
    list: Best set of medoids (indices of medoid points in the dataset).
    """

    def calculate_cost(old_medoids: list, new_medoids: list) -> float:
        """Calculate the total cost of the medoid change."""

        i = set(old_medoids).difference(set(new_medoids)).pop()
        h = set(new_medoids).difference(set(old_medoids)).pop()
        
        current_fitting = fit(old_medoids)

        old_medoids_reduced = old_medoids[:]
        old_medoids_reduced.remove(i)
        reduced_fitting = fit(old_medoids_reduced)

        total_cost = 0
        for j, Oj in enumerate(data):
            closest_medoid = current_fitting[j]

            d_ji = np.linalg.norm(np.array(Oj) - np.array(data[i]))
            d_jh = np.linalg.norm(np.array(Oj) - np.array(data[h]))

            if closest_medoid == i:
                j2 = reduced_fitting[j]
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(data[j2]))
                
                # Case 1: Point moves from Oi to a previously existing medoid
                if d_jh >= d_jj2:
                    total_cost += d_jj2 - d_ji

                # Case 2: Point moves from Oi to the new medoid
                else:
                    total_cost += d_jh - d_ji
            else:
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(data[closest_medoid]))
                
                # Case 4: Point moves from other than Oi to the new medoid
                if d_jh < d_jj2:
                    total_cost += d_jh - d_jj2

                # Case 3: Point does not change its cluster - cost does not change

        return total_cost
    
    def fit(medoids: list) -> list:
        """Returns list of medoid indices for each element in dataset."""
        return [find_closest_medoid(item, medoids) for item in data]
    
    def find_closest_medoid(point: tuple, medoids: list) -> int:
        """
        Determine the closest medoid for a given point.

        Parameters:
        point (tuple): The data point to classify.
        medoids (list): Indices of medoids.

        Returns:
        int: Index of the closest medoid.
        """

        closest_medoid = None
        min_distance = float("inf")

        for i in medoids:
            distance = np.linalg.norm(np.array(point) - np.array(data[i]))
            if distance < min_distance:
                min_distance = distance
                closest_medoid = i

        return closest_medoid

    def get_neighbors(current: list) -> list:
        """
        Generate neighbors by swapping one medoid with a non-medoid point.

        Parameters:
        current (list): Indices of current set of medoids.

        Returns:
        list: List of neighbors (each neighbor is a set of medoids).
        """

        neighbors = []
        non_medoids = [idx for idx in range(len(data)) if idx not in set(current)]

        for medoid in current:
            for non_medoid in non_medoids:
                neighbor = current[:]
                neighbor[neighbor.index(medoid)] = non_medoid
                neighbors.append(neighbor)

        return neighbors

    def calculate_quality(medoids: list) -> float:
        return 0
    
    bestnode = None
    best_cost = float("inf")

    for _ in range(numlocal):
        # Step 1: Choose a random set of medoids
        current = random.sample(range(len(data)), num_clusters)

        # Step 2: Reset neighbor count
        j = 0

        # Iterate over neighbors in the graph
        neighbors = get_neighbors(current)
        while j < maxneighbor and len(neighbors) > 0:
            # Step 3: Calculate the cost of medoids swap with random neighbor
            neighbor = random.choice(neighbors)
            neighbors.remove(neighbor)

            cost = calculate_cost(current, neighbor)

            # Step 4: Optimization step
            if cost < 0:
                j = 0
                current = neighbor
                neighbors = get_neighbors(current)
            else:
                j += 1

        # Step 5: Update the best node if the current node is better
        final_cost = calculate_quality(current)
        if final_cost < best_cost:
            bestnode = current
            best_cost = final_cost

    # Step 6: Return the best medoids indices found
    return bestnode

# Example usage:
data = [
    (1.0, 2.0), (1.5, 1.8), (5.0, 8.0), (8.0, 8.0), (1.0, 0.6), (9.0, 11.0), (8.0, 2.0), (10.0, 2.0), (9.0, 3.0)
]

num_clusters = 2
numlocal = 2
maxneighbor = max(250, 0.0125 * num_clusters * (len(data) - num_clusters))

best_medoids_idx = clarans(data, num_clusters, numlocal, maxneighbor)
best_medoids = [data[i] for i in best_medoids_idx]

print("Best medoids (points):", best_medoids)

# Tworzymy wykres
data = np.array(data)
best_medoids = np.array(best_medoids)

plt.scatter(data[:, 0], data[:, 1], color='blue', label='Punkty danych')  # Punkty danych
plt.scatter(best_medoids[:, 0], best_medoids[:, 1], color='red', marker='X', s=100, label='Medoidy')  # Medoidy

# Opcjonalnie, dodanie etykiet i tytułu
plt.title("Wykres punktów danych oraz medoidów")
plt.xlabel("X")
plt.ylabel("Y")

# Dodanie legendy
plt.legend()

# Wyświetlenie wykresu
plt.grid(True)
plt.show()