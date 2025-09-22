import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('edge.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image 'edge.jpg' not found in the current directory!")

#Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.absolute(sobel))

# Prewitt
kernelx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
kernely = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])

prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)
prewitt = cv2.magnitude(np.float32(prewittx), np.float32(prewitty))
prewitt = np.uint8(np.absolute(prewitt))

#Canny
canny = cv2.Canny(img, 100, 200)   

titles = ['Original Image', 'Sobel', 'Prewitt', 'Canny']
images = [img, sobel, prewitt, canny]

plt.figure(figsize=(10,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
