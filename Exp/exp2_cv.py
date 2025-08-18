import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image = cv2.imread("image2.jpg")  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
M = np.ones(image.shape, dtype="uint8") * 50

brighter = cv2.add(image, M)

darker = cv2.subtract(image, M)

rows, cols, _ = image.shape

scaled = cv2.resize(image, None, fx=0.5, fy=0.5)

translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(image, translation_matrix, (cols, rows))

rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))

titles = ["Original", "Brighter", "Darker", "Scaled", "Translated", "Rotated"]
images = [image, brighter, darker, scaled, translated, rotated]

plt.figure(figsize=(10, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.show()
