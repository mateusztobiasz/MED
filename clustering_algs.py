from sklearn_extra.cluster import CLARA, KMedoids

class ClusteringAlgs:
    @staticmethod
    def pam(data, clusters):
        kmedoids = KMedoids(n_clusters=clusters, method="pam", init="build").fit(data)

        return kmedoids.cluster_centers_, kmedoids.labels_

    @staticmethod
    def clara(data, clusters):
        clara = CLARA(n_clusters=clusters, init="build").fit(data)

        return clara.cluster_centers_, clara.labels_
