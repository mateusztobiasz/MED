import argparse
import datetime
import random
import numpy as np

from arff_parser import ArffParser
from clustering_algs import ClusteringAlgs
from plot_drawer import PlotDrawer

random.seed(42)


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
        medoid_indexes = fit(medoids)
        best_medoids = [data[i] for i in medoid_indexes]

        np_best_medoids = np.array(best_medoids)
        np_data = np.array(data)

        return np.linalg.norm(np_best_medoids - np_data)

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
            clusters = fit(current)
            bestnode = current
            best_cost = final_cost

    # Step 6: Return the best medoids indices found
    return bestnode, clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster data from a file.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--clusters", type=int, required=True, help="Number of clusters.")

    args = parser.parse_args()

    num_clusters = args.clusters
    file_path = args.filepath

    parser = ArffParser()
    drawer = PlotDrawer()

    parser.load_file(file_path)
    data = parser.get_data()

    

    # PAM
    start = datetime.datetime.now()
    best_medoids, clusters = ClusteringAlgs.pam(data, num_clusters)
    end = datetime.datetime.now()   

    print(f"[PAM] Time: {end - start}")
    print("[PAM]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)

    # CLARA
    start = datetime.datetime.now()
    best_medoids, clusters = ClusteringAlgs.clara(data, num_clusters)
    end = datetime.datetime.now()

    print(f"[CLARA] Time: {end - start}")
    print("[CLARA]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)

    # CLARANS
    numlocal = 2
    maxneighbor = max(250, 0.0125 * num_clusters * (len(data) - num_clusters))

    start = datetime.datetime.now()
    best_medoids_idx, clusters = clarans(data, num_clusters, numlocal, maxneighbor)
    end = datetime.datetime.now()
    best_medoids = [data[i] for i in best_medoids_idx]

    print(f"[CLARANS] Time: {end - start}")
    print("[CLARANS]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)
