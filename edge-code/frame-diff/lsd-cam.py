import cv2
import os
import numpy as np
from math import pi


def frame_diff(prev, curr, kernel=np.array((5, 5), dtype=np.uint8),
               filter_size=3, blocksize=11, C=3):
    #frame-difference
    frame_diff = cv2.subtract(curr, prev)
    frame_diff = cv2.medianBlur(frame_diff, filter_size)
    #apply thresholding
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blocksize, C)
    mask = cv2.medianBlur(mask, filter_size)
    #Morphology close - close up the dots from F-difference: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask

#obtain first MHI
def MHI_1(prev_mhi, frame_diff, tau=255, delta=32):
    next_MHI = np.zeros_like(prev_mhi).astype(int)

    next_MHI[frame_diff == 255] = tau
    next_MHI[frame_diff != 255] = np.clip(prev_mhi[frame_diff != 255].astype(int) - delta, 0, 255)

    return np.clip(next_MHI, 0, 255).astype(np.uint8)

###The above is the MHI equation

(MHI1_f, 1/3, 2/3, 1, 0, 1, 0.5, pow=1)
def pseudocolor_generator(X, v1, v2, v3, v1_node, v2_node, v3_node, pow=1):
    I = X.astype(float) / 255.

    s1 = np.clip(1 - ((1. / ((v1_node - v1) ** 2)) * (v1 - I) ** 2) ** pow, 0, 1)
    s2 = np.clip(1 - ((1. / ((v2_node - v2) ** 2)) * (v2 - I) ** 2) ** pow, 0, 1)
    s3 = np.clip(1 - ((1. / ((v3_node - v3) ** 2)) * (v3 - I) ** 2) ** pow, 0, 1)

    return np.stack([s1, s2, s3], axis=2)



# Capture video from the first camera connected to your computer
cap = cv2.VideoCapture(0)  # 0 is the index of the first camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    os.exit(0)

prev = None
MHI1_f = None
of_hsv = None

# Loop to continuously get frames from the camera
while True:
    # Read a new frame
    ret, rgb_frame = cap.read()

    # ret
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    if MHI1_f is None and prev is not None:
        MHI1_f = np.zeros_like(prev)

    if of_hsv is None:
        of_hsv = np.zeros_like(rgb_frame)

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    if prev is not None:
        _frame_diff = frame_diff(prev, frame)
        MHI1_f = MHI_1(MHI1_f, _frame_diff, tau=255, delta=32)
        pseudocolor5 = pseudocolor_generator(MHI1_f, 1/3, 2/3, 1, 0, 1, 0.5, pow=1)


        # Display the frame
        #cv2.imshow('frame_diff', _frame_diff)
        cv2.imshow("frame-diff", _frame_diff)
        cv2.imshow("MHI", MHI1_f)
        cv2.imshow("pseudocolor5", pseudocolor5)

    prev = frame

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()