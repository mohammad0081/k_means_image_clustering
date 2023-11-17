# K Means Image Clustering
* Implementing K-Means Algorithm to cluster images of numbers 1 to 5

K-means clustering is a type of unsupervised learning algorithm used to group similar data points together in a dataset. It is commonly used in image segmentation to partition an image into multiple segments based on the similarity of the pixels. The goal of image segmentation is to change the representation of an image into something that is more meaningful and easier to analyze. It is usually used for locating objects and creating boundaries 1.

**How Does It Work**

1- Choose the number of clusters you want to find which is k.

2- Randomly assign the data points to any of the k clusters.

3- Then calculate the center of the clusters.

4- Calculate the distance of the data points from the centers of each of the clusters.

5- Depending on the distance of each data point from the cluster, reassign the data points to the nearest clusters.

6- Again calculate the new cluster center.

7- Repeat steps 4,5 and 6 till data points don’t change the clusters, or till we reach the assigned number of iterations

in our project and dataset, because Images are 2D matrices, we sould flatten them as a 1D vector that we can implement our algorithm on it


**Libraries**

1- Numpy

2-Pillow

3-Numba
