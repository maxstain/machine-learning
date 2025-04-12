from src.utils.public_imports import *


class ClusteringAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()

    def kmeans_clustering(self, data, n_clusters=5):
        """Perform K-means clustering"""
        scaled_data = self.scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        return clusters, kmeans
