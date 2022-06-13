from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = r'E:\MiniProj\Image_segmentation\1.jpeg'
image = cv2.imread(path)
image = cv2.resize(image, (256,256))
image = np.array(image)


np.random.seed(44)
num_items = 1000
color_array = np.random.choice(range(0, 256,3), 3*num_items).reshape(-1,3)

num_classes = 10
kmeans_model = KMeans(n_clusters = num_classes)
kmeans_model.fit(color_array)


label_class = kmeans_model.predict(image.reshape(-1,3)).reshape(256,256)
plt.imshow(label_class)
plt.show()