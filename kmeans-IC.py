import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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


def init_centroids(X, k):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:k]]
    return centroids


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K) :
        points = X[idx == k]

        if len(points) > 0:
            centroids[k] = np.mean(points, axis= 0)

    return centroids


def find_closest_centroid(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = []
        for j in range(centroids.shape[0]):
            norm_i_j = np.linalg.norm(X[i] - centroids[j])
            distances.append(norm_i_j)

        idx[i] = np.argmin(distances)

    return idx


def run_k_means(X, initial_centroids, max_iters=100):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print(f'Epoch : {i} | K = {k}')
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

dataset = data_redaer()

for k in [3, 4, 5, 6, 7] :

    initial_centroids = init_centroids(dataset, k)
    # Run k-means clustering
    centroids, labels = run_k_means(dataset, initial_centroids)

    # Reshape the centroids into images
    centroids = centroids.reshape((k, 16, 16))

    if k == 5 :
        for item in centroids :
            plt.imshow(item, cmap='gray')
            plt.show()

    for i in range(k):

        image_data = np.uint8(centroids[i])
        image = Image.fromarray(image_data)

        image.save(f"centroids/{k}/centroid{i + 1}.png")
