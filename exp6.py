import cv2
import numpy as np
import matplotlib.pyplot as plt

def correlation(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    img_float = img.astype(np.float32)
    kernel_float = kernel.astype(np.float32)

    padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img_float)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel_float)

    return output

def convolution(img, kernel):

    flipped_kernel = np.flipud(np.fliplr(kernel))
    return correlation(img, flipped_kernel)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - min_val) * (255.0 / (max_val - min_val))
    return np.clip(scaled, 0, 255).astype(np.uint8)


candidate_paths = [
    "straw.jpg",
    "image.jpeg",
    "image.jpg",
    "image.png",
    "OIP.png",
    "OIP2.png",
    "Figure_1.png",
]

img = None
for path in candidate_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        break

if img is None:
    raise FileNotFoundError(
        "No input image found. Tried: " + ", ".join(candidate_paths)
    )

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)

correlated_img = correlation(img, kernel)
convoluted_img = convolution(img, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Correlation")
plt.imshow(normalize_to_uint8(correlated_img), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Convolution")
plt.imshow(normalize_to_uint8(convoluted_img), cmap="gray")
plt.axis("off")

plt.show()