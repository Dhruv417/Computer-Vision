import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("straw.jpg", cv2.IMREAD_GRAYSCALE)

negative = 255 - img

brightness = cv2.add(img, 50)

contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

gamma = 2.2
gamma_corrected = np.array(255 * (img / 255) ** (1/gamma), dtype='uint8')

titles = ['Original', 'Negative', 'Brightness +50', 'Contrast x1.5', 'Threshold', 'Gamma (2.2)']
images = [img, negative, brightness, contrast, threshold, gamma_corrected]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
