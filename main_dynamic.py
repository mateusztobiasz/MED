import argparse
import datetime
import random
import numpy as np

import sys, os
sys.path.append(os.getcwd())

from arff_parser import ArffParser
from clustering_algs import ClusteringAlgs
from plot_drawer import PlotDrawer

random.seed(42)

class KMedoids:

    current_fitting: list

    def __init__(self, data: list, num_clusters: int):
        """
        Parameters:
        data (list of tuple of floats): Dataset to cluster.
        num_clusters (int): Number of clusters.
        """
        
        self.data = data
        self.num_clusters = num_clusters

    def calculate_cost(self, old_medoids: list, new_medoids: list) -> tuple:
        """Calculate the total cost of the medoid change."""

        i = set(old_medoids).difference(set(new_medoids)).pop()
        h = set(new_medoids).difference(set(old_medoids)).pop()

        new_fitting = self.current_fitting[:]

        old_medoids_reduced = old_medoids[:]
        old_medoids_reduced.remove(i)

        total_cost = 0
        for j, Oj in enumerate(self.data):
            closest_medoid = self.current_fitting[j]

            d_ji = np.linalg.norm(np.array(Oj) - np.array(self.data[i]))
            d_jh = np.linalg.norm(np.array(Oj) - np.array(self.data[h]))

            if closest_medoid == i:
                j2 = self.find_closest_medoid(self.data[j], old_medoids_reduced)
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(self.data[j2]))

                # Case 1: Point moves from Oi to a previously existing medoid
                if d_jh >= d_jj2:
                    total_cost += d_jj2 - d_ji
                    new_fitting[j] = j2

                # Case 2: Point moves from Oi to the new medoid
                else:
                    total_cost += d_jh - d_ji
                    new_fitting[j] = h
            else:
                d_jj2 = np.linalg.norm(np.array(Oj) - np.array(self.data[closest_medoid]))

                # Case 4: Point moves from other than Oi to the new medoid
                if d_jh < d_jj2:
                    total_cost += d_jh - d_jj2
                    new_fitting[j] = h

                # Case 3: Point does not change its cluster - cost does not change

        return total_cost, new_fitting

    def fit(self, medoids: list) -> list:
        """Returns list of medoid indices for each element in dataset."""
        return [self.find_closest_medoid(item, medoids) for item in self.data]

    def find_closest_medoid(self, point: tuple, medoids: list) -> int:
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
            distance = np.linalg.norm(np.array(point) - np.array(self.data[i]))
            if distance < min_distance:
                min_distance = distance
                closest_medoid = i

        return closest_medoid

    def calculate_quality(self, medoids: list) -> float:
        medoid_indexes = self.fit(medoids)
        best_medoids = [data[i] for i in medoid_indexes]

        np_best_medoids = np.array(best_medoids)
        np_data = np.array(self.data)

        return np.linalg.norm(np_best_medoids - np_data)

    def get_neighbors(self, current: list) -> list:
        """
        Generate neighbors by swapping one medoid with a non-medoid point.

        Parameters:
        current (list): Indices of current set of medoids.

        Returns:
        list: List of neighbors (each neighbor is a set of medoids).
        """

        neighbors = []
        non_medoids = [idx for idx in range(len(self.data)) if idx not in set(current)]

        for medoid in current:
            for non_medoid in non_medoids:
                neighbor = current[:]
                neighbor[neighbor.index(medoid)] = non_medoid
                neighbors.append(neighbor)

        return neighbors

    def clarans(self, numlocal, maxneighbor) -> list:
        """
        CLARANS clustering algorithm operating on an abstract graph.

        Parameters:
        numlocal (int): Number of local minima attempts.
        maxneighbor (int): Maximum number of neighbors to explore.

        Returns:
        list: Best set of medoids (indices of medoid points in the dataset).
        """

        bestnode = None
        best_cost = float("inf")

        for _ in range(numlocal):
            # Step 1: Choose a random set of medoids
            current = random.sample(range(len(self.data)), self.num_clusters)
            self.current_fitting = self.fit(current)

            # Step 2: Reset neighbor count
            j = 0

            # Iterate over neighbors in the graph
            neighbors = self.get_neighbors(current)
            while j < maxneighbor and len(neighbors) > 0:
                # Step 3: Calculate the cost of medoids swap with random neighbor
                neighbor = random.choice(neighbors)
                neighbors.remove(neighbor)

                cost, neighbor_fitting = self.calculate_cost(current, neighbor)

                # Step 4: Optimization step
                if cost < 0:
                    j = 0
                    current = neighbor
                    self.current_fitting = neighbor_fitting
                    neighbors = self.get_neighbors(current)
                else:
                    j += 1

            # Step 5: Update the best node if the current node is better
            final_cost = self.calculate_quality(current)
            if final_cost < best_cost:
                clusters = self.fit(current)
                bestnode = current
                best_cost = final_cost

        # Step 6: Return the best medoids indices found
        return bestnode, clusters

    def pam(self) -> tuple:
        """
        PAM clustering algorithm.

        Returns:
        list: Best set of medoids (indices of medoid points in the dataset).
        """

        current = random.sample(range(len(self.data)), self.num_clusters)
        self.current_fitting = self.fit(current)

        optimized = True
        while optimized:
            neighbors = self.get_neighbors(current)
            optimized = False
            for neighbor in neighbors:
                cost, neighbor_fitting = self.calculate_cost(current, neighbor)

                if cost < 0:
                    optimized = True
                    current = neighbor
                    self.current_fitting = neighbor_fitting
                    neighbors = self.get_neighbors(current)
                    break

        clusters = self.fit(current)
        return current, clusters

    def clara(self, n_samples: int, sample_size: int) -> list:
        data = list(range(len(self.data)))
        sample = random.sample(data, sample_size)
        
        best_medoids = None
        best_quality = float("inf")

        for _ in range(n_samples):
            kmedoids = KMedoids([data[s] for s in sample], self.num_clusters)
            medoids, _ = kmedoids.pam()
            medoids = [sample[m] for m in medoids]

            quality = self.calculate_quality(medoids)
            if quality < best_quality:
                best_medoids = medoids
                best_quality = quality

            s = random.sample(data, sample_size - self.num_clusters)
            s.extend(best_medoids)
            sample = list(set(s))
        
        return best_medoids, self.fit(best_medoids)

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

    
    kmedoids = KMedoids(data, num_clusters)

    # PAM
    start = datetime.datetime.now()
    best_medoids_idx, clusters = kmedoids.pam()
    end = datetime.datetime.now()   
    best_medoids = [data[i] for i in best_medoids_idx]

    print(f"[PAM] Time: {end - start}")
    print("[PAM]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)

    # CLARA
    start = datetime.datetime.now()
    best_medoids_idx, clusters = kmedoids.clara(5, 40 + 2 * num_clusters)
    end = datetime.datetime.now()
    best_medoids = [data[i] for i in best_medoids_idx]

    print(f"[CLARA] Time: {end - start}")
    print("[CLARA]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)

    # CLARANS
    numlocal = 2
    maxneighbor = max(250, 0.0125 * num_clusters * (len(data) - num_clusters))

    start = datetime.datetime.now()
    best_medoids_idx, clusters = kmedoids.clarans(numlocal, maxneighbor)
    end = datetime.datetime.now()
    best_medoids = [data[i] for i in best_medoids_idx]

    print(f"[CLARANS] Time: {end - start}")
    print("[CLARANS]: Best medoids (points):", best_medoids)
    drawer.draw(data, best_medoids, clusters)
