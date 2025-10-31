import cv2
import numpy as np

IMAGE_PATH = "exp10.png"

THRESHOLD = 10  

seed_selected = False
seed_point = None
original_img = None

def on_mouse(event, x, y, flags, param):
    global seed_selected, seed_point
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        seed_selected = True
        print("Seed selected at:", seed_point)

def region_growing(img, seed, threshold):
    height, width = img.shape[:2]

    segmented = np.zeros((height, width), np.uint8)
    visited = np.zeros_like(segmented)

    seed_value = img[seed[1], seed[0]]
    stack = [seed]

    while stack:
        x, y = stack.pop()

        if visited[y, x] == 1:
            continue
        visited[y, x] = 1

        if abs(int(img[y, x]) - int(seed_value)) < threshold:
            segmented[y, x] = 255

            # 4-connected neighbors
            neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            for nx, ny in neighbors:
                if 0 <= nx < width and 0 <= ny < height:
                    stack.append((nx, ny))

    return segmented

if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH, 0)  # Directly load grayscale image

    if img is None:
        print("Image not found! Check the path or filename.")
        exit()

    original_img = img.copy()

    cv2.imshow("Input Image", img)
    cv2.setMouseCallback("Input Image", on_mouse)

    print("Click on the image to select a seed point... Press ESC to exit.")

    while True:
        cv2.imshow("Input Image", original_img)

        if seed_selected:
            segmented = region_growing(img, seed_point, THRESHOLD)
            cv2.imshow("Segmented Output", segmented)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
