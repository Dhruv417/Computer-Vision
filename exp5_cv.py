import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale for simplicity)
img = cv2.imread("straw.jpg", cv2.IMREAD_GRAYSCALE)

# --- Point operations ---
# 1. Negative Image
negative = 255 - img

# 2. Brightness adjustment (+50)
brightness = cv2.add(img, 50)

# 3. Contrast stretching (alpha=1.5, beta=0)
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# 4. Thresholding (binary)
_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 5. Gamma correction (γ = 2.2)
gamma = 2.2
gamma_corrected = np.array(255 * (img / 255) ** (1/gamma), dtype='uint8')

# --- Display results ---
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
