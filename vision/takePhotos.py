# Take 10 photos of the chessboard and save them
import cv2
import os
import sys

# Path to the folder where the photos will be saved
path = "vision/chessboard_webcam"

# Create the folder if it does not exist
if not os.path.exists(path):
    os.makedirs(path)

def main(argv):
 # Initialize the webcam
 cap = cv2.VideoCapture(0)
 i = 0

 while True:
  ret, frame = cap.read()

  cv2.imshow("Press 'c' to take a photo", frame)

  key = cv2.waitKey(30)
  if key == ord('c') or key == 99:
   print("Photo " + str(i+1) + " taken.")
   cv2.imwrite(path + "/photo" + str(i) + ".jpg", frame)
   i += 1
  if key == ord('q') or key == 27:
   break
  if i == 10:
   break

if __name__ == "__main__":
 main(sys.argv[1:])