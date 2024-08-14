import torch
import torch.nn.functional as F
import numpy as np

# from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from models.cm import ClusterMemory


class MemoryTable(object):
    def __init__(self, cluster, num_features, temp, momentum, mode='cm'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cluster = cluster
        try:
            self.num_clusters = cluster.n_clusters
            self.num_clusters_cur = cluster.n_clusters
        except:
            self.num_clusters = 2000
            self.num_clusters_cur = 2000
        # self.cluster_memory = ClusterMemory(2048, self.num_clusters, temp=temp,
        #                                     momentum=momentum, use_hard=use_hard).to(self.device)
        self.cluster_memory = ClusterMemory(num_features, self.num_clusters, temp=temp,
                                            momentum=momentum, mode=mode).to(self.device)
        self.pseudo_labels = []

    def update_labels(self, features, outlier=True, mat_dist=None, min_instances=None, camera_indexes=None, kmeans=False):
        if kmeans:
            self.cluster.fit(features.numpy())
        else:
            if camera_indexes is not None:
                self.cluster.fit(features.numpy(), camera_indexes.numpy(), mat_dist, outlier)
            else:
                self.cluster.fit(features.numpy(), mat_dist, outlier)

        labels = self.cluster.labels_
        centroids = self.cluster.cluster_centers_
        num_clusters = self.cluster.n_clusters

        # remove the cluster which labeled number < min_instances
        if min_instances is not None:
            labels_index, labels_count = np.unique(labels, return_counts=True)
            removed_labels = labels_index[np.where(labels_count < min_instances)]
            retained_labels = labels_index[np.where(labels_count >= min_instances)]
            retained_labels = sorted(retained_labels[retained_labels != -1])
            centroids = centroids[retained_labels]
            offset_table = np.zeros_like(labels)
            for removed_label in sorted(removed_labels):
                if removed_label == -1:
                    continue
                offset_table[labels == removed_label] = 0
                offset_table[labels > removed_label] -= 1
                labels[labels == removed_label] = -1
            labels += offset_table
            num_clusters -= removed_labels.size

        self.pseudo_labels = labels
        self.num_clusters_cur = num_clusters
        self.num_clusters = num_clusters
        self.cluster_memory.features = torch.tensor(centroids, dtype=torch.float32, device=self.device)
        self.cluster_memory.features = F.normalize(self.cluster_memory.features, dim=1)
