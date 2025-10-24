# requirements opencv and numpy
# pip install opencv-python
# pip install numpy

import cv2
import numpy as np

# Globals
clicked_points = []
click_down = None
is_dragging = False
orig_img = None
display_img = None

def click_event(event, x, y, flags, param):
    """Handle mouse events to select points or detect drag."""
    global clicked_points, click_down, is_dragging, display_img

    if event == cv2.EVENT_LBUTTONDOWN:
        click_down = (x, y)
        is_dragging = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON and click_down is not None:
            dx = x - click_down[0]
            dy = y - click_down[1]
            if abs(dx) > 5 or abs(dy) > 5:
                is_dragging = True

    elif event == cv2.EVENT_LBUTTONUP:
        if click_down is not None and not is_dragging:
            if len(clicked_points) < 4:
                clicked_points.append((x, y))
                cv2.circle(display_img, (x, y), 10, (0, 255, 0), -1)
                cv2.imshow("Image", display_img)
                print(f"Points selected: {len(clicked_points)}")
                if len(clicked_points) == 4:
                    print("Press 'c' to crop and save the image.")
        click_down = None

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

if __name__ == "__main__":
    file_path = "inputs/Pago.jpg"
    orig_img = cv2.imread(file_path)
    if orig_img is None:
        print("Error: could not load image file. Check the path.")
        exit(1)

    display_img = orig_img.copy()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    h, w = display_img.shape[:2]
    max_dim = 800
    cv2.resizeWindow("Image", min(w, max_dim), min(h, max_dim))
    cv2.imshow("Image", display_img)
    cv2.setMouseCallback("Image", click_event)

    print("Click four document corners (no drag).")
    print("Press 'e' to erase last point, 'r' to reset, 'c' to crop & save, 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_points.clear()
            display_img = orig_img.copy()
            cv2.imshow("Image", display_img)
            print("Reset. Points selected: 0")
        elif key == ord('e'):
            if clicked_points:
                clicked_points.pop()
                display_img = orig_img.copy()
                for pt in clicked_points:
                    cv2.circle(display_img, pt, 10, (0, 255, 0), -1)
                cv2.imshow("Image", display_img)
                print(f"Erased last point. Points selected: {len(clicked_points)}")
        elif key == ord('c') and len(clicked_points) == 4:
            warped = four_point_transform(orig_img, clicked_points)
            filename = "cropped_output.png"
            cv2.imwrite(filename, warped)
            print(f"Cropped image saved as '{filename}'")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
