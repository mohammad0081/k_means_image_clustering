from PIL import Image
import numpy as np


def data_redaer():
    dataset = []
    for i in range(5):
        img = Image.open(f'images/usps_{i+1}.jpg')

        img_array = np.array(img)

        for i in range(0 ,img_array.shape[0] , 16):
            for j in range(0, img_array.shape[1] , 16):

                small_image = img_array[i : i + 16, j : j + 16]
                small_image = small_image.flatten()

                dataset.append(small_image)

    dataset = np.array(dataset)

    return dataset


def assign_labels(dataset, centroids):

    distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis= 1)

    return labels


def update_centroids(dataset, labels, K):

    n_features = dataset.shape[1]
    updated_centroids = np.zeros((k, n_features))

    for i in range(K):
        cluster_points = dataset[ labels == i]

        if len(cluster_points) > 0:
            updated_centroids[i] = np.mean(cluster_points, axis=0)

    return updated_centroids


def k_means_image_clustering(dataset, k, init_centroids, max_iters=100):
    centroids = init_centroids

    for i in range(max_iters):
        print(f'Epoch : {i} | K = {k}')

        labels = assign_labels(dataset, centroids)
        updated_centroids = update_centroids(dataset, labels, k)

        centroids = updated_centroids

    return centroids, labels




dataset = data_redaer()

for k in [3, 4, 5, 6, 7] :

    init_centroids = dataset[ np.random.choice(len(dataset), k) ]
    # Run k-means clustering
    centroids, labels = k_means_image_clustering(dataset, k, init_centroids= init_centroids)

    # Reshape the centroids into images
    centroids = centroids.reshape((k, 16, 16))

    for i in range(k):

        image_data = np.uint8(centroids[i])
        image = Image.fromarray(image_data)

        image.save(f"centroids/{k}/centroid{i + 1}.png")



