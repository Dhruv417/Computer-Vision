import cv2
import numpy as np
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

print("Please upload an image file:")
uploaded = files.upload()

# Get the uploaded file name
filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale for better morphological operations
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Perform Erosion
eroded = cv2.erode(gray, kernel, iterations=1)

# Perform Dilation
dilated = cv2.dilate(gray, kernel, iterations=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Grayscale image
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Grayscale Image', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# Eroded image
axes[1, 0].imshow(eroded, cmap='gray')
axes[1, 0].set_title('Eroded Image', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

# Dilated image
axes[1, 1].imshow(dilated, cmap='gray')
axes[1, 1].set_title('Dilated Image', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print("\nOperations completed successfully!")
print(f"Original image shape: {img.shape}")
print(f"Kernel size: {kernel.shape}")