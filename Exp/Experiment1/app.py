# Importing required libraries
from PIL import Image
import cv2
import numpy as np


pil_img = Image.open("German.png")  
pil_img.show(title="PIL Original Image")

pil_array = np.array(pil_img)
print("PIL Image converted to Array:\n", pil_array)   


resized_pil = pil_img.resize((100, 100))
resized_pil.show(title="Resized Image (100x100)")

cv_img = cv2.imread("German.png")  
cv2.imshow("Original Image (OpenCV)", cv_img)


gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()