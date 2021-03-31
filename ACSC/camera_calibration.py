import numpy as np
import cv2
import glob
import yaml
from checkerboard import detect_checkerboard
import os 
#import pathlib

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/user/ACSC/calibration_data/**/**/*.png')

# path = 'results'
# pathlib.Path(path).mkdir(parents=True, exist_ok=True)

found = 0
for fname in images:  # Here, 10 can be changed to whatever number you like to choose
    orig = cv2.imread(fname) # Capture frame-by-frame
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # Find the chess board corners
    corners, score  = detect_checkerboard(img, (7,5))

    # If found, add object points, image points (after refining them)
    if corners is not None:
        corners = corners.astype(np.float32)
        objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
        imgpoints.append(corners)
        # Draw and display the corners\
        #print(len(corners), score, corners.shape)
        #rint(orig.shape)
        #image = cv2.drawChessboardCorners(orig, (7,5), corners, True)
        found += 1
        # cv2.imshow('image', image)
        # cv2.waitKey(500)
        # if you want to save images with detected corners
        # uncomment following 2 lines and lines 5, 18 and 19
        # image_name = path + '/calibresult' + str(found) + '.png'
        # cv2.imwrite(image_name, img)

print("Number of images used for calibration: ", found)

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

camera_matrix = np.asarray(mtx)
dist_coeff = np.asarray(dist).flatten()

print('camera_matrix', camera_matrix)
print('dist_coeff', dist_coeff)

data_fldr = os.listdir('/home/user/ACSC/calibration_data')
assert len(data_fldr) == 1, "There should be exactly one folder of image/clouds from collect-calibrate, found {}".format(data_fldr)
data_root = os.path.join('/home/user/ACSC/calibration_data', data_fldr[0])

intrinsic_file_content = ""
for row in camera_matrix:
    for col in row:
        intrinsic_file_content += "{} ".format(col)
    intrinsic_file_content += "\n"

with open(os.path.join(data_root, 'intrinsic'), "w") as f:
    f.write(intrinsic_file_content)

distortion_file_content = ""
for coeff in dist_coeff:
    distortion_file_content += "{} ".format(coeff)

with open(os.path.join(data_root, 'distortion'), "w") as f:
    f.write(distortion_file_content)

# transform the matrix and distortion coefficients to writable lists
# data = {'camera_matrix': np.asarray(mtx).tolist(),
#         'dist_coeff': np.asarray(dist).tolist()}

# and save it to a file
# with open("calibration_matrix.yaml", "w") as f:
#     yaml.dump(data, f)

# done
