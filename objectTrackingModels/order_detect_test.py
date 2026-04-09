import cv2

# test image:
img = cv2.imread("objectTrackingModels/testImages/frame_0188.png")
cv2.imwrite("debug_crop.png", img[0:140, 8:126])

frame = cv2.cvtColor(img[0:140, 8:126], cv2.COLOR_BGR2GRAY)

for label, path in [(1, "templates/fish_order.png"), (2, "templates/shrimp_order.png")]:
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    print(f"{'fish' if label == 1 else 'shrimp'}: score = {max_val:.3f}")