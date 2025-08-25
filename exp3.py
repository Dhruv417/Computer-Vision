import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('exp3image.jpg', cv2.IMREAD_GRAYSCALE)

hist_eq = cv2.equalizeHist(img)

min_val = np.min(img)
max_val = np.max(img)
contrast_stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(hist_eq, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(contrast_stretched, cmap='gray')
plt.title("Contrast Stretching")
plt.axis("off")

# Plot histograms
plt.subplot(2,3,4)
plt.hist(img.ravel(), bins=256, range=[0,256])
plt.title("Original Histogram")

plt.subplot(2,3,5)
plt.hist(hist_eq.ravel(), bins=256, range=[0,256])
plt.title("Histogram Equalization Histogram")

plt.subplot(2,3,6)
plt.hist(contrast_stretched.ravel(), bins=256, range=[0,256])
plt.title("Contrast Stretching Histogram")

plt.tight_layout()

plt.show(block=False)
plt.pause(100)   # keeps plot open for 3 seconds
plt.close()