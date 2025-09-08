import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA to 8-bit grayscale."""
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image shape")
    return g.astype(np.uint8)

def digital_negative(img: np.ndarray) -> np.ndarray:
    return 255 - img

def threshold(img: np.ndarray, T: int, mode: str = "binary") -> np.ndarray:
    modes = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "truncate": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    _, out = cv2.threshold(img, T, 255, modes[mode])
    return out

def clipping(img: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(img, low, high).astype(np.uint8)

def bit_plane_slicing(img: np.ndarray) -> list:
    planes = []
    for k in range(8):
        plane = ((img >> k) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes

def intensity_level_slicing(img: np.ndarray, r1: int, r2: int, keep_background: bool = False) -> np.ndarray:
    mask = (img >= r1) & (img <= r2)
    if keep_background:
        out = img.copy()
        out[mask] = 255
    else:
        out = np.zeros_like(img)
        out[mask] = 255
    return out

def demo_show(img: np.ndarray):
    neg = digital_negative(img)
    thr = threshold(img, T=128, mode="binary")
    clip = clipping(img, low=50, high=200)
    planes = bit_plane_slicing(img)
    ils_binary = intensity_level_slicing(img, r1=80, r2=160, keep_background=False)
    ils_keep = intensity_level_slicing(img, r1=80, r2=160, keep_background=True)

    plt.figure(figsize=(13, 10))
    plt.subplot(3, 4, 1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
    plt.subplot(3, 4, 2); plt.imshow(neg, cmap='gray'); plt.title("Digital Negative"); plt.axis('off')
    plt.subplot(3, 4, 3); plt.imshow(thr, cmap='gray'); plt.title("Threshold (T=128)"); plt.axis('off')
    plt.subplot(3, 4, 4); plt.imshow(clip, cmap='gray'); plt.title("Clipping [50,200]"); plt.axis('off')

    for i, p in enumerate(planes):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(p, cmap='gray')
        plt.title(f"Bit Plane {i}")
        plt.axis('off')

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1); plt.imshow(ils_binary, cmap='gray'); plt.title("Intensity Level Slice\n[80,160], binary"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(ils_keep, cmap='gray'); plt.title("Intensity Level Slice\n[80,160], keep bg"); plt.axis('off')
    plt.show()

def main():
    img_name = "exp4 image.jpg"
    if Path(img_name).exists():
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = to_gray(img)
    else:
        print(f"⚠️ '{img_name}' not found, using synthetic gradient.")
        x = np.linspace(0, 255, 256, dtype=np.uint8)
        img = np.tile(x, (256, 1))

    demo_show(img)

if __name__ == "__main__":
    main()
