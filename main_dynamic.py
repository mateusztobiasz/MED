import random
import numpy as np

def clarans_graph(data, num_clusters, numlocal, maxneighbor):
    """
    CLARANS clustering algorithm operating on an abstract graph.

    Parameters:
    data (list of list of floats): Dataset to cluster.
    num_clusters (int): Number of clusters.
    numlocal (int): Number of local minima attempts.
    maxneighbor (int): Maximum number of neighbors to explore.

    Returns:
    list: Best set of medoids (indices of medoid points in the dataset).
    """
    def calculate_cost(medoids):
        """Calculate the total cost of the current medoids."""
        cost = 0
        for point in data:
            min_dist = float("inf")
            for medoid in medoids:
                dist = np.linalg.norm(np.array(point) - np.array(data[medoid]))
                min_dist = min(min_dist, dist)
            cost += min_dist
        return cost
    
    
    def find_closest_medoid(point, medoids, data):
        """
        Determine the closest medoid for a given point.

        Parameters:
        point (list): The data point to classify.
        medoids (list): Indices of the medoids.
        data (list of list of floats): Dataset to compare distances.

        Returns:
        int: Index of the closest medoid.
        """
        closest_medoid = None
        min_distance = float("inf")

        for medoid in medoids:
            distance = np.linalg.norm(np.array(point) - np.array(data[medoid]))
            if distance < min_distance:
                min_distance = distance
                closest_medoid = medoid

        return closest_medoid

    def get_neighbors(current):
        """
        Generate neighbors by swapping one medoid with a non-medoid point.

        Parameters:
        current (list): Current set of medoids.

        Returns:
        list: List of neighbors (each neighbor is a set of medoids).
        """
        neighbors = []
        current_set = set(current)
        non_medoids = [idx for idx in range(len(data)) if idx not in current_set]

        for medoid in current:
            for non_medoid in non_medoids:
                neighbor = current[:]
                neighbor[neighbor.index(medoid)] = non_medoid
                neighbors.append(neighbor)

        return neighbors

    bestnode = None
    best_cost = float("inf")

    for _ in range(numlocal):
        # Step 1: Choose a random set of medoids
        current = random.sample(range(len(data)), num_clusters)

        # Step 2: Reset neighbor count
        j = 0

        # Iterate over neighbors in the graph
        neighbors = get_neighbors(current)
        while j < maxneighbor:
            # Step 3: Calculate the cost of medoids swap with random neighbor
            neighbor = random.choice(neighbors)
            current_cost = calculate_cost(current)
            neighbor_cost = calculate_cost(neighbor)

            # Step 4: Optimization step
            if neighbor_cost < current_cost:
                j = 0 
                current = neighbor
                neighbors = get_neighbors(current)
            else:
                j += 1

        # Step 5: Update the best node if the current node is better
        final_cost = calculate_cost(current)
        if final_cost < best_cost:
            bestnode = current
            best_cost = final_cost

    # Step 6: Return the best medoids found
    return bestnode

# Example usage:
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
]

num_clusters = 3
numlocal = 5
maxneighbor = 10

best_medoids = clarans_graph(data, num_clusters, numlocal, maxneighbor)
print("Best medoids (indices):", best_medoids)
print("Best medoids (points):", [data[idx] for idx in best_medoids])
