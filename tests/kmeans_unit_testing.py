import unittest
import numpy as np
from clustering_algorithms.kmeans import k_means_clustering
from sklearn.cluster import KMeans


class TestKmeans(unittest.TestCase):
    def test_simple(self):
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 9], [8, 8]])
        MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles = k_means_clustering(data=data,
                                                                                                            no_dim=2,
                                                                                                            feature_importance=None,
                                                                                                            no_centroids=len(
                                                                                                                data))
        self.assertEqual(MSE, 0)
        self.assertEqual(len(clusters), len(set(clusters)))

    def test_simple_3d(self):
        data = np.random.random((3, 4))
        MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles = k_means_clustering(data=data,
                                                                                                            no_dim=4,
                                                                                                            feature_importance=None,
                                                                                                            no_centroids=2)
        kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, tol=0.0000001).fit(data)
        assert np.allclose(kmeans.labels_, clusters)
        assert np.allclose(kmeans.cluster_centers_, centroids)

    def test_medium_data_medium_dim_3(self):
        no_dim, no_clusters, no_pts = 4, 4, 200
        data = np.random.random((no_pts, no_dim))
        MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles = k_means_clustering(
            data=data, no_dim=no_dim,
            feature_importance=True,
            no_centroids=no_clusters)
        kmeans = KMeans(n_clusters=no_clusters, init=initial_centroids, n_init=1, tol=0.0000001).fit(data)
        assert np.allclose(kmeans.labels_, clusters)
        assert np.allclose(kmeans.cluster_centers_, centroids)

    def test_large_data_large_dim(self):
        no_dim, no_clusters, no_pts = 6, 5, 2000
        data = np.random.random((no_pts, no_dim))
        MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles = k_means_clustering(
            data=data, no_dim=no_dim,
            feature_importance=True,
            no_centroids=no_clusters)
        kmeans = KMeans(n_clusters=no_clusters, init=initial_centroids, n_init=1, tol=0.0000001).fit(data)
        assert np.allclose(kmeans.labels_, clusters)
        assert np.allclose(kmeans.cluster_centers_, centroids)

    def test_sparse_data_large_dim(self):
        no_dim, no_clusters, no_pts = 6, 5, 20
        data = np.random.random((no_pts, no_dim))
        MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles = k_means_clustering(
            data=data, no_dim=no_dim,
            feature_importance=True,
            no_centroids=no_clusters)
        kmeans = KMeans(n_clusters=no_clusters, init=initial_centroids, n_init=1, tol=0.0000001).fit(data)
        assert np.allclose(kmeans.labels_, clusters)
        assert np.allclose(kmeans.cluster_centers_, centroids)


if __name__ == '__main__':
    unittest.main()
