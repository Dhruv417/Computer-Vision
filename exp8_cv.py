import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image 
img = cv2.imread('die.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Image not found. Make sure 'input.jpg' is in the same directory as this script.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the kernel
kernel = np.ones((5, 5), np.uint8)

# Perform dilation and erosion
dilated = cv2.dilate(gray, kernel, iterations=1)
eroded = cv2.erode(gray, kernel, iterations=1)

# Display the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Dilated")
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Eroded")
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
