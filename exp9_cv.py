import cv2
import numpy as np

img = cv2.imread('edge.jpg', cv2.IMREAD_GRAYSCALE)

# Binary Image
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Kernel for morphological operations
kernel = np.ones((5,5), np.uint8)

# Opening and Closing
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Resize all images
img = cv2.resize(img, (300, 300))
binary = cv2.resize(binary, (300, 300))
opening = cv2.resize(opening, (300, 300))
closing = cv2.resize(closing, (300, 300))

# Add label text on each image
cv2.putText(img, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
cv2.putText(binary, "Binary", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
cv2.putText(opening, "Opening", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
cv2.putText(closing, "Closing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

# Combine result
top_row = np.hstack((img, binary))
bottom_row = np.hstack((opening, closing))
combined = np.vstack((top_row, bottom_row))

cv2.imshow('Morphological Operations - Opening and Closing', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
