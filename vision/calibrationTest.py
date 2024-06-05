import numpy as np
import cv2 as cv
import glob
 
# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chessboard size
n = 7
m = 9
 
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n*m,3), np.float32)
objp[:,:2] = np.mgrid[0:n,0:m].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('vision/chessboard_webcam/*.jpg')
 
for fname in images:
 img = cv.imread(fname)
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
 # Find the chess board corners
 ret, corners = cv.findChessboardCorners(gray, (n,m), None)
 
 # If found, add object points, image points (after refining them)
 if ret == True:
  objpoints.append(objp)
 
  corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
  imgpoints.append(corners)
 
 # Draw and display the corners
 cv.drawChessboardCorners(img, (n,m), corners, ret)
 cv.imshow('img', img)
 key = cv.waitKey(500)
 if key == ord('n') or key == 110:
  continue
 if key == ord('q') or key == 27:
  exit()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the results
print("RMS: ", ret)
print("Camera matrix: [[", mtx[0][0], ", ", mtx[0][1], ", ", mtx[0][2], "], [", mtx[1][0], ", ", mtx[1][1], ", ", mtx[1][2], "], [", mtx[2][0], ", ", mtx[2][1], ", ", mtx[2][2], "]]")
print("Distortion coefficients: \n", dist)
 
cv.destroyAllWindows()