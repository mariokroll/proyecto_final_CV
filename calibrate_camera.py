import cv2
import imageio
import copy
import numpy as np

path = 'pictures/picture{}.png'

def load_images(filenames):
    return [cv2.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    array = [[0 + i*dx, 0 + j*dy, 0] for i in range(chessboard_shape[0]) for j in range(chessboard_shape[1])]
    return array

def calibrate_camera():
    imgs = load_images([path.format(i) for i in range(5, 25)])
    size = (8, 6)

    sq_size = 15
    objp = np.zeros((np.prod(size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0: size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= sq_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    object_points = []
    image_points = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            image_points.append(corners2)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def save_parameters(mtx, dist):
    np.save('mtx.npy', mtx)
    np.save('dist.npy', dist)


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    save_parameters(mtx, dist)
