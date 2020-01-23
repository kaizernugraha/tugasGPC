import cv2
import numpy as np

gambar = cv2.imread('kaizer1.jpg')
np.array(gambar, dtype=np.uint8)
gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
gambar_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

# canny edge proses
gambar_canny = cv2.Canny(gambar, 100, 200)

# sobel proses
gambar_sobelx = cv2.Sobel(gambar_gaussian, cv2.CV_8U, 1, 0, ksize=5)
gambar_sobely = cv2.Sobel(gambar_gaussian, cv2.CV_8U, 0, 1, ksize=5)
gambar_sobel = gambar_sobelx + gambar_sobely


# prewitt proses
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
gambar_prewittx = cv2.filter2D(gambar_gaussian, -1, kernelx)
gambar_prewitty = cv2.filter2D(gambar_gaussian, -1, kernely)


# tampilkan
cv2.imshow("Original Image", gambar)
cv2.imshow("Canny", gambar_canny)
cv2.imshow("Sobel X", gambar_sobelx)
cv2.imshow("Sobel Y", gambar_sobely)
cv2.imshow("Sobel", gambar_sobel)
cv2.imshow("Prewitt X", gambar_prewittx)
cv2.imshow("Prewitt Y", gambar_prewitty)
cv2.imshow("Prewitt", gambar_prewittx + gambar_prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()
