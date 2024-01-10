import cv2


def show_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"X: {x}, Y: {y}")


image = cv2.imread("./1/1-book1.jpg")
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_pixel_coordinates)
while True:
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
