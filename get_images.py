import cv2
from picamera2 import Picamera2
import os

def take_picture(picam: Picamera2):
    return picam.capture_array()

def initialize_camera():
    picam = Picamera2()
    picam.preview_configuration.main.size=(250, 250)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    return picam

def save_pictures(picture, direct, number):
    cv2.imwrite(direct + "/picture{}.png".format(number), picture)

def take_pictures(n: int, picam: Picamera2):
    if not os.path.exists("pictures"):
        os.makedirs("pictures")
    taken = 0
    t = 0
    while True:
        frame = take_picture(picam)
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        t += 1
        
        if t % 600 == 0:
            save_pictures(frame, "pictures", taken)
            taken += 1
            if taken >= n:
                break

if __name__ == "__main__":
   picam = initialize_camera()
   take_pictures(25, picam)
   cv2.destroyAllWindows()
