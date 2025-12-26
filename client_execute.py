import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def calculate_distance(cluster1, cluster2):
  
    return np.linalg.norm(cluster1['center'] - cluster2)

def client_kmeans(local_data, server_clusters=None, k=4, random_seed=None):
    if not server_clusters:
        return perform_kmeans(local_data, k, random_seed)
    else:
        return adjust_clusters_with_radius(local_data, server_clusters, k, random_seed)


def perform_kmeans(data, num_clusters, random_seed=None):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_seed, n_init='auto').fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    clusters = {i: [] for i in range(num_clusters)}  
    for idx, label in enumerate(labels):  
        clusters[label].append(data[idx])

    cluster_info = []  
    for i in range(num_clusters):
        cluster_center = centers[i]
        cluster_points = clusters[i]
        cluster_info.append({'center': cluster_center, 'num_points': len(cluster_points)})
    return cluster_info


def adjust_clusters_with_radius(local_data, server_clusters, k, random_seed=None):
    labels = []

    for point in local_data:
        min_dist = float('inf') 
        closest_cluster_index = None 
        for i, cluster in enumerate(server_clusters):
            dist = np.linalg.norm(point - cluster['center'])
            if dist < min_dist:
                min_dist = dist
                closest_cluster_index = i
        labels.append(closest_cluster_index)

    updated_clusters = {i: [] for i in range(k)}
    for point, label in zip(local_data, labels):
        updated_clusters[label].append(point)

    new_clusters = []
    total_distance_all_clusters = 0  

    t_dist = 0

    for i in range(k):
        if updated_clusters[i]:
            for poin in updated_clusters[i]:
                distan = calculate_distance(server_clusters[i], poin)
                t_dist += distan

    for i in range(k):
        if updated_clusters[i]: 
            cluster_points = np.array(updated_clusters[i]) 
            center = np.mean(cluster_points, axis=0) 
            num_points = len(cluster_points)  

            distances = cdist(cluster_points, [center], 'euclidean')  
            total_distance = np.sum(distances) 

            avg_radius = np.mean(distances)  

            new_clusters.append({
                'center': center,
                'num_points': num_points,
                'avg_radius': avg_radius
            })

            total_distance_all_clusters += total_distance

    return new_clusters, t_dist

