import cv2
import numpy as np

# img = cv2.imread("kamera1.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("kamera1.jpg")

# img_resized = cv2.resize(img, None, fx=2, fy=1)
# img_flipped = cv2.flip(img, 0)
# img[10:50:, 40:60:, :] = [255, 255, 0]
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_hsv[10:50:, 40:60:, :] = [0, 255, 255]
# img_rgb_changed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

# cv2.imwrite("gray.jpg", img)

# cv2.imshow("Slika s automobilom", img)
# cv2.imshow("Slika promijenjena", img_rgb_changed)
# cv2.imshow("Hsv slika", img_hsv)


# img = cv2.imread("hand.png", cv2.IMREAD_GRAYSCALE)

# thres, binary_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

# img = cv2.imread("kamera1.jpg")
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# donja_zelena_granica = np.array([75, 80, 50])
# gornja_zelena_granica = np.array([88, 255, 255])

# mask = cv2.inRange(img_hsv, donja_zelena_granica, gornja_zelena_granica)
# filtered_img = cv2.bitwise_and(img_hsv, img, mask=mask)
# filtered_img_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2BGR)

# cv2.imshow("Binarna slika", binary_image)
# cv2.imshow("Slika s automobila", img)
# cv2.imshow("Mask",mask)
# cv2.imshow("Filtrirana slika", filtered_img_bgr)

# img = np.zeros((400, 400), dtype = np.uint8)
# img[20:160, 40:120] = 255

# rect = cv2.boundingRect(img)
# cv2.rectangle(img, rect, color=(100, 2, 5), thickness=2)

# print(rect)

# cv2.imshow("slika", img)


# img = cv2.imread("kamera1.jpg")
# img_blurred = cv2.blur(img, (5, 5))
# img_Gblurred = cv2.GaussianBlur(img, (3, 3), 0)

# cv2.imshow("Zamucena slika", img_Gblurred)


# img = cv2.imread("kamera2.jpg", cv2.IMREAD_GRAYSCALE)
# img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

# sobel_x = cv2.Sobel(img_blurred, cv2.CV_16S, 1, 0)
# sobel_y = cv2.Sobel(img_blurred, cv2.CV_16S, 0, 1)
# sobel_x_abs = cv2.convertScaleAbs(sobel_x)
# sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# cv2.imshow("sobel x", sobel_x_abs)
# cv2.imshow("sobel y", sobel_y_abs)
# cv2.imshow("slika", img)

img = cv2.imread("kamera2.jpg", cv2.IMREAD_GRAYSCALE)


cv2.waitKey()
cv2.destroyAllWindows()