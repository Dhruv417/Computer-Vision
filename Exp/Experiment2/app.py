import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("German.jpg")
img2 = cv2.imread("Dog.jpg")

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

add_img = cv2.add(img1, img2)
sub_img = cv2.subtract(img1, img2)
mul_img = cv2.multiply(img1, img2)
div_img = cv2.divide(img1, img2)

and_img = cv2.bitwise_and(img1, img2)
or_img = cv2.bitwise_or(img1, img2)
xor_img = cv2.bitwise_xor(img1, img2)
not_img = cv2.bitwise_not(img1)

brightness = cv2.convertScaleAbs(img1, alpha=1, beta=50)  
contrast = cv2.convertScaleAbs(img1, alpha=1.5, beta=0)    
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

titles = ['Original', 'Added', 'Subtracted', 'Multiplied', 'Divided',
          'AND', 'OR', 'XOR', 'NOT', 'Brightness', 'Contrast', 'Threshold']

images = [img1, add_img, sub_img, mul_img, div_img,
          and_img, or_img, xor_img, not_img, brightness, contrast, threshold]

plt.figure(figsize=(15,10))
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    if len(images[i].shape) == 2: 
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
