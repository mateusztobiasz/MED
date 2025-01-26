import random
import numpy as np
import networkx as nx
from itertools import combinations

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
    list: Best set of medoids.
    """

    def calculate_cost(old_medoids: tuple, new_medoids: tuple) -> float:
        """Calculate the total cost of the medoid change."""

        Oi = set(old_medoids).difference(set(new_medoids)).pop()
        Oh = set(new_medoids).difference(set(old_medoids)).pop()
        
        current_fitting = fit(list(old_medoids))

        old_medoids_list = list(old_medoids)
        old_medoids_list.remove(Oi)
        reduced_fitting = fit(old_medoids_list)

        total_cost = 0
        for j, Oj in enumerate(data):
            closest_medoid = old_medoids[current_fitting[j]]

            d_ji = np.linalg.norm(np.array(Oj) - np.array(Oi))
            d_jh = np.linalg.norm(np.array(Oj) - np.array(Oh))

            if closest_medoid == Oi:
                Oj2 = old_medoids_list[reduced_fitting[j]]
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(Oj2))
                
                # Case 1: Point moves from Oi to a previously existing medoid
                if d_jh >= d_jj2:
                    total_cost += d_jj2 - d_ji

                # Case 2: Point moves from Oi to the new medoid
                else:
                    total_cost += d_jh - d_ji
            else:
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(closest_medoid))
                
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
        medoids (tuple): Medoids.

        Returns:
        int: Index of the closest medoid.
        """

        closest_medoid = None
        min_distance = float("inf")

        for i in range(len(medoids)):
            distance = np.linalg.norm(np.array(point) - np.array(medoids[i]))
            if distance < min_distance:
                min_distance = distance
                closest_medoid = i

        return closest_medoid
    
    def create_graph() -> nx.Graph:
        """
        Create a graph where nodes are sets of k medoids and edges connect nodes
        that differ by exactly one medoid.

        Parameters:
        data (list of list of floats): Dataset to cluster.
        num_clusters (int): Number of clusters (size of each medoid set).

        Returns:
        networkx.Graph: Graph representation of the medoid space.
        """
        G = nx.Graph()

        # Generate all possible sets of k medoids (nodes of the graph)
        nodes = list(combinations(data, num_clusters))

        # Add nodes to the graph
        for node in nodes:
            G.add_node(node)

        # Add edges between nodes that differ by exactly one medoid
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                if len(set(node1).difference(set(node2))) == 1:  # Differ by one medoid
                    G.add_edge(node1, node2)

        return G

    def calculate_quality(medoids: tuple) -> float:
        return 0

    bestnode = None
    best_cost = float("inf")

    graph = create_graph()

    for _ in range(numlocal):
        # Step 1: Choose a random node in the graph as the starting point
        current = random.choice(list(graph.nodes))

        # Step 2: Initialize neighbor count
        j = 0

        neighbors = list(graph.neighbors(current))
        while j < maxneighbor and len(neighbors) > 0:
            # Step 3: Calculate the cost of current swap with random neighbor
            neighbor = random.choice(neighbors)
            neighbors.remove(neighbor)

            cost = calculate_cost(current, neighbor)

            # Step 4: Optimization step
            if cost < 0:
                current = neighbor
                neighbors = list(graph.neighbors(current))
                j = 0
            else:
                j += 1

        # Step 5: Update the best node if the current node is better
        final_cost = calculate_quality(current)
        if final_cost < best_cost:
            bestnode = current
            best_cost = final_cost

    # Step 6: Return the best medoids found
    return bestnode

# Example usage:
data = [
    (1.0, 2.0), (1.5, 1.8), (5.0, 8.0), (8.0, 8.0), (1.0, 0.6), (9.0, 11.0), (8.0, 2.0), (10.0, 2.0), (9.0, 3.0)
]

num_clusters = 2
numlocal = 2
maxneighbor = max(250, 0.0125 * num_clusters * (len(data) - num_clusters))

best_medoids = clarans(data, num_clusters, numlocal, maxneighbor)

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