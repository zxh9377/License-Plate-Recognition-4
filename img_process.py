import cv2

src = cv2.imread('carplate.jpg')
src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(src, cv2.CV_8U, 1, 0,  ksize = 3)
ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
# 膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)
# 腐蚀一次，去掉细节
erosion = cv2.erode(dilation, element1, iterations = 1)
# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2,iterations = 3)

cv2.imshow("test",binary)
cv2.waitKey(0)