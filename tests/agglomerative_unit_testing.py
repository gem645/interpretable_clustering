import unittest
import numpy as np
from clustering_algorithms.agglomerative_clustering import agglomerative_clustering_complete_link, agglomerative_clustering_single_link
from sklearn.cluster import AgglomerativeClustering


class TestAgglomerative(unittest.TestCase):
    def calculate_single_linkage(self, data):
        cluster_labels, distances, joined_clusters, feature_importance = agglomerative_clustering_single_link(data=data,
                                                                                                              return_feature_importance=True)
        agglom = AgglomerativeClustering(compute_distances=True, linkage="single").fit(data)
        assert np.allclose(agglom.distances_, distances)

    def calculate_complete_linkage(self, data):
        cluster_labels, distances, joined_clusters, feature_importance = agglomerative_clustering_complete_link(data=data,
                                                                                                              return_feature_importance=True)
        agglom = AgglomerativeClustering(compute_distances=True, linkage="complete").fit(data)
        assert np.allclose(agglom.distances_, distances)


    def test_simple(self):
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 9], [8, 8]])
        self.calculate_single_linkage(data)
        self.calculate_complete_linkage(data)

    def test_3_dimensional_small_dataset(self):
        data = np.random.random((20, 3))
        self.calculate_single_linkage(data)
        self.calculate_complete_linkage(data)

    def test_3_dimensional_large_dataset(self):
        data = np.random.random((200, 3))
        self.calculate_single_linkage(data)
        self.calculate_complete_linkage(data)

    def test_5_dimensional_medium_dataset(self):
        data = np.random.random((80, 5))
        self.calculate_single_linkage(data)
        self.calculate_complete_linkage(data)

    def test_6_dimensional(self):
        data = np.random.random((20, 6))
        self.calculate_single_linkage(data)
        self.calculate_complete_linkage(data)


if __name__ == '__main__':
    unittest.main()
