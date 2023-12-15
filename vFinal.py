import cv2
import numpy as np
from picamera2 import Picamera2
from enum import Enum
import time

class Estados(Enum):
    PATRON1 = 0
    PATRON2 = 1
    PATRON3 = 2
    TRACKER = 3

def initialize_camera():
    picam = Picamera2()
    picam.preview_configuration.main.size=(250, 250)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    return picam

def take_picture(picam: Picamera2):
    return picam.capture_array()


def detect_contours(frame, color_boundaries):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_boundaries[0], color_boundaries[1])
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def detect_square(frame, color_boundaries):
    contours = detect_contours(frame, color_boundaries)
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            epsilon = 0.01*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                squares.append(approx)
    return squares


def detect_triangle(frame, color_boundaries):
    contours = detect_contours(frame, color_boundaries)
    triangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            epsilon = 0.01*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3:
                triangles.append(approx)
    return triangles


def detect_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    if circles is not None:
        circles = list(np.round(circles[0, :]).astype("int"))
        return max(circles, key = lambda x: x[2])
    return None


def apply_mask_yellow(image, color_boundaries):
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    lower_yellow = color_boundaries[0]
    upper_yellow = color_boundaries[1]

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    parts = cv2.bitwise_and(image, image, mask=mask)
    return parts


def detect_yellow_triangle(frame):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    approx = detect_triangle(frame, [lower_yellow, upper_yellow])
    return approx


def detect_yellow_square(frame):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    approx = detect_square(frame, [lower_yellow, upper_yellow])
    return approx


def detect_yellow_circle(frame):
    frame2 = frame.copy()
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    circle = detect_circle(frame2)
    frame2 = apply_mask_yellow(frame2, [lower_blue, upper_blue])
    return circle, frame2


def tracker(frame, fgbg):
    fgmask = fgbg.apply(frame)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([20, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    fgmask = cv2.bitwise_and(fgmask, mask)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x + w/2), int(y + h/2))
    return frame, center

def main():
    picam = initialize_camera()
    estado = Estados.PATRON1
    counterF = {'T': 0, 'S': 0, 'C': 0}
    counter = 0
    fgbg = cv2.createBackgroundSubtractorMOG2()
    draws = []
    while True:
        frame = take_picture(picam)
        if estado == Estados.PATRON1:
            triangle = detect_yellow_triangle(frame)
            square = detect_yellow_square(frame)
            circle, _ = detect_yellow_circle(frame)
            if triangle == [] and square == [] and circle is None:
                counter = 0
                counterF = {'T': 0, 'S': 0, 'C': 0}
            else:
                counter += 1
                if triangle != []:
                    counterF['T'] += 1
                    cv2.drawContours(frame, triangle, -1, (0, 255, 0), 3)
                if square != []:
                    counterF['S'] += 1
                if circle is not None:
                    counterF['C'] += 1
                if counter >= 40:
                    if counterF['T'] > counterF['S'] and counterF['T'] > counterF['C']:
                        print('Primer elemento detectado')
                        estado = Estados.PATRON2
                        a = time.time()
                    counter = 0
                    counterF = {'T': 0, 'S': 0, 'C': 0}
        elif estado == Estados.PATRON2:
            if time.time() - a > 3:
                if time.time() - a > 15:
                    print('Patron 2 no detectado. Volviendo a patrón 1')
                    estado = Estados.PATRON1
                triangle = detect_yellow_triangle(frame)
                square = detect_yellow_square(frame)
                circle, _ = detect_yellow_circle(frame)
                if triangle == [] and square == [] and circle is None:
                    counter = 0
                    counterF = {'T': 0, 'S': 0, 'C': 0}
                else:
                    counter += 1
                    if triangle != []:
                        counterF['T'] += 1
                    if square != []:
                        counterF['S'] += 1
                        cv2.drawContours(frame, square, -1, (0, 255, 0), 3)
                    if circle is not None:
                        counterF['C'] += 1
                    if counter >= 40:
                        if counterF['S'] > counterF['T'] and counterF['S'] > counterF['C']:
                            print('Segundo elemento detectado')
                            estado = Estados.PATRON3
                            a = time.time()
                        else:
                            counter = 0
                            counterF = {'T': 0, 'S': 0, 'C': 0}
                            print('Patron 2 no detectado. Volviendo a patrón 1')
                            estado = Estados.PATRON1
        elif estado == Estados.PATRON3:
            if time.time() - a > 3:
                if time.time() - a > 15:
                    print('Patron 2 no detectado. Volviendo a patrón 1')
                    estado = Estados.PATRON1
                triangle = detect_yellow_triangle(frame)
                square = detect_yellow_square(frame)
                circle, frame2 = detect_yellow_circle(frame)
                if triangle == [] and square == [] and circle is None:
                    counter = 0
                    counterF = {'T': 0, 'S': 0, 'C': 0}
                else:
                    counter += 1
                    if triangle != []:
                        counterF['T'] += 1
                    if square != []:
                        counterF['S'] += 1
                    if circle is not None:
                        counterF['C'] += 1
                        x, y, r = circle
                        if frame2[x][y].all() != 0:
                            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                    if counter >= 40:
                        if counterF['C'] > counterF['T'] and counterF['C'] > counterF['S']:
                            print('Tercer patrón detectado. Habilitando el tracker')
                            estado = Estados.TRACKER
                        else:
                            counter = 0
                            counterF = {'T': 0, 'S': 0, 'C': 0}
                            print('Patron 3 no detectado. Volviendo a patrón 1')
                            estado = Estados.PATRON1
        elif estado == Estados.TRACKER:
            frame, center = tracker(frame, fgbg)
            if center != None:
                draws.append(center)
            for c in draws:
                cv2.circle(frame, c, 5, (0, 0, 255), 1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    picam.stop()

if __name__ == "__main__":
    main()
