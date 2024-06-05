import sys
import cv2 as cv
import numpy as np

# Parameters
hsv_parameters = [80, 95, 100, 255, 60, 215] # Color filter parameters
hough_params = [50, 10, 15, 100] # Hough Circle Transform parameters [param1, param2, minRadius, maxRadius]
erodeI = 1 # Erode iterations
dilateI = 3 # Dilate iterations
blurI = 3 # Blur iterations
max_lenC = 5 # Center memory length for Mean Filter
max_lenR = 10 # Radius memory length for Mean Filter
ballD = 100 # Ball diameter [mm]
camera_matrix = [[ 447.25075629538344 ,  0.0 ,  313.8308823686259 ], [ 0.0 ,  446.82063878930535 ,  239.2343871475514 ], [ 0.0 ,  0.0 ,  1.0 ]]

# Mean Filter variables
center_memory = [None] * max_lenC
radius_memory = [None] * max_lenR
lenC = 0
lenR = 0
j = 0
k = 0

# Camera variables
f_x = camera_matrix[0][0] # Focal length in x
f_y = camera_matrix[1][1] # Focal length in y
f = (f_x + f_y) / 2 # Focal length
numD = ballD * f # Numerator of the distance calculation

def filter(center, radius):
 global center_memory
 global radius_memory
 global lenC
 global lenR
 global j
 global k
 resultC = (0, 0)
 resultR = 0
 acumC1 = 0
 acumC2 = 0
 acumR = 0

 print(j)
 center_memory[j] = center
 radius_memory[k] = radius

 if lenC < max_lenC:
  lenC = lenC + 1

 if lenR < max_lenR:
  lenR = lenR + 1

 for i in range(lenC):
  acumC1 += center_memory[i][0]
  acumC2 += center_memory[i][1]
 resultC = (int(acumC1 / lenC), int(acumC2 / lenC))
 j = j + 1

 for i in range(lenR):
  acumR += radius_memory[i]
 resultR = int(acumR / lenR)
 k = k + 1

 if j >= max_lenC:
  j = 0

 if k >= max_lenR:
  k = 0

 return resultC, resultR


def circleDetection(frame):
 
 # Initialize the circle parameters
 filter_center = (0, 0)
 filter_radius = 0

 # Preprocess the frame
 frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 frame_threshold = cv.inRange(frame_HSV, (hsv_parameters[0], hsv_parameters[2], hsv_parameters[4]), (hsv_parameters[1], hsv_parameters[3], hsv_parameters[5]))
 frame_erode = cv.erode(frame_threshold, None, iterations=erodeI)
 frame_dilate = cv.dilate(frame_erode, None, iterations=dilateI)
 frame_blur = cv.medianBlur(frame_dilate, blurI)
 
 # Apply the Hough Circle Transform
 rows = frame_blur.shape[0]
 circles = cv.HoughCircles(frame_blur, cv.HOUGH_GRADIENT, 1, rows / 8,
 param1=hough_params[0], param2=hough_params[1],
 minRadius=hough_params[2], maxRadius=hough_params[3])
 # Draw the circles
 if circles is not None:
  circles = np.uint16(np.around(circles))
  for i in circles[0, :]:
   # Circle center and radius
   center = (i[0], i[1])
   radius = i[2]
   # Filter the center and radius
   filter_center, filter_radius = filter(center, radius)
   # Draw the circle
   cv.circle(frame, filter_center, 1, (0, 100, 100), 3)
   cv.circle(frame, filter_center, filter_radius, (255, 0, 255), 3)
 
 return frame, frame_blur, filter_center, filter_radius

def center2circle(frame, center):
 if(center != (0, 0)):
  # Calculate center distance
  (h, w, _) = frame.shape
  im_center = (w//2, h//2)
  # Draw the center and the distance
  cv.circle(frame, im_center, 1, (0, 100, 100), 3)
  cv.arrowedLine(frame, center, im_center, (0, 0, 255), 3)
 return frame

def camera2circle(frame, radius):
 if(radius != 0):
  # Calculate the camera distance
  distance = numD / (2*radius) # Distance in [mm]
  # Print the distance
  cv.putText(frame, "Distance: " + str(distance) + " mm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
 return frame

def main(argv):
 
 # Start capturing the video
 cap = cv.VideoCapture(0)

 while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  if frame is None:
   print("Error: no frame.")
   break
  # Apply the circle detection
  frame, frame_treshold, center, radius = circleDetection(frame)
  # Calculate distance from circle to center
  frame = center2circle(frame, center)
  # Calculate distance from camera to circle
  distance = camera2circle(frame, radius)

  # Display the resulting frame
  cv.imshow('Circle Detection', frame)
  cv.imshow('Circle Detection Treshold', frame_treshold)

  key = cv.waitKey(30)
  if key == ord('q') or key == 27:
   break
 return 0

if __name__ == "__main__":
 main(sys.argv[1:])