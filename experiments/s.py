import cv2
import os
import numpy as np
from math import pi


def frame_diff(prev, curr, kernel=np.array((5, 5), dtype=np.uint8),
               filter_size=3, blocksize=11, C=3):
    frame_diff = cv2.subtract(curr, prev)
    frame_diff = cv2.medianBlur(frame_diff, filter_size)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blocksize, C)
    mask = cv2.medianBlur(mask, filter_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask


def frame_diff2(prev, curr, kernel=np.array((5, 5), dtype=np.uint8),
                filter_size=3, filter_size_post=3, blocksize=15, C=3):
    frame_diff = cv2.subtract(curr, prev)
    frame_diff = cv2.medianBlur(frame_diff, filter_size)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blocksize, C)
    mask = cv2.medianBlur(mask, filter_size_post)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def MHI_1(prev_mhi, frame_diff, tau=255, delta=32):
    next_MHI = np.zeros_like(prev_mhi).astype(int)

    next_MHI[frame_diff == 255] = tau
    next_MHI[frame_diff != 255] = np.clip(prev_mhi[frame_diff != 255].astype(int) - delta, 0, 255)

    return np.clip(next_MHI, 0, 255).astype(np.uint8)


def MHI_pseudocolor(curr_MHI):
    I = curr_MHI.astype(float) / 255.

    #phi_r = 1/5 - 1/2 * pi
    #phi_g = 1/5 - 1/2 * pi - 13/14
    #phi_b = 1/5 - 1/2 * pi - 6/14

    C_r = 0.5 * (1 + np.cos(4 * pi / 3. * I))
    C_g = 0.5 * (1 + np.cos(4 * pi / 3 * I - 2 / 3 * pi))
    C_b = 0.5 * (1 + np.cos(4 * pi / 3 * I - 4 / 3 * pi))

    ret = np.stack([C_r, C_g, C_b], axis=2)

    return ret


def MHI_pseudocolor2(curr_MHI):
    I = curr_MHI.astype(float) / 255.

    #phi_r = 1/5 - 1/2 * pi
    #phi_g = 1/5 - 1/2 * pi - 13/14
    #phi_b = 1/5 - 1/2 * pi - 6/14

    C_r = np.cos(4 * pi / 3. * I) ** 2
    C_g = np.cos(4 * pi / 3 * I - 2 / 3 * pi) ** 2
    C_b = np.cos(4 * pi / 3 * I - 4 / 3 * pi) ** 2

    ret = np.stack([C_r, C_g, C_b], axis=2)

    return ret


def MHI_pseudocolor3(curr_MHI):
    I = curr_MHI.astype(float) / 255.

    # 0.5**2 - (I - 0.5)**2 = (0.5 - I + 0.5) * (0.5 + I - 0.5)
    #                         (1 - I ) *
    #phi_r = 1/5 - 1/2 * pi
    #phi_g = 1/5 - 1/2 * pi - 13/14
    #phi_b = 1/5 - 1/2 * pi - 6/14

    C_r = I ** 2
    C_g = (4 * I * (1 - I)) ** 4
    C_b = I ** 0.25

    ret = np.stack([C_b, C_g, C_r], axis=2)

    return ret


def MHI_pseudocolor4(curr_MHI):
    I = curr_MHI.astype(float) / 255.

    v1, v2, v3 = 1 / 3, 2 / 3, 1.

    C_r = np.clip((1 - 9 * (v1 - I) ** 2), 0, 1)
    C_g = np.clip((1.0 - 9 * (v2 - I) ** 2), 0, 1)
    C_b = np.clip((1.0 - 4 * (v3 - I) ** 2), 0, 1)
    ret = np.stack([C_r, C_g, C_b], axis=2)

    return ret


def MHI_pseudocolor_fail(curr_MHI):
    I = curr_MHI.astype(float) / 255.
    f_I = 0.25 + 0.75 * I

    phi_r = 1 / 5 - 1 / 2 * pi
    phi_g = 1 / 5 - 1 / 2 * pi - 13 / 14
    phi_b = 1 / 5 - 1 / 2 * pi - 6 / 14

    C_r = f_I * (np.sin(2 * pi * (-I + phi_r) * 0.5 + 0.5))
    C_g = f_I * (np.sin(2 * pi * (-I + phi_g) * 0.5 + 0.5))
    C_b = f_I * (np.sin(2 * pi * (-I + phi_b) * 0.5 + 0.5))

    ret = np.stack([C_b, C_g, C_r], axis=2)

    return ret


def MHI_pseudocolor_fail2(curr_MHI):
    I = curr_MHI.astype(float) / 255.
    f_I = 0.25 + 0.75 * I

    phi_r = (1 / 5 - 1 / 2) * pi
    phi_g = (1 / 5 - 1 / 2 - 13 / 14) * pi
    phi_b = (1 / 5 - 1 / 2 - 6 / 14) * pi

    C_r = f_I * (np.sin(pi * -I + phi_r * 0.5 + 0.5 * pi))
    C_g = f_I * (np.sin(pi * -I + phi_g * 0.5 + 0.5 * pi))
    C_b = f_I * (np.sin(pi * -I + phi_b * 0.5 + 0.5 * pi))

    ret = np.stack([C_b, C_g, C_r], axis=2)

    return ret


def MHI_springtime(curr_MHI):
    H = 0.5 - 0.5 * np.sqrt(1 - curr_MHI ** 2)
    S = 0.8 * np.ones_like(curr_MHI)
    V = np.log2(curr_MHI + 1.)

    ret = np.stack([179 * H, 255 * S, 255 * V], axis=2).astype(np.uint8)

    return cv2.cvtColor(ret, cv2.COLOR_HSV2RGB)


def pseudocolor_generator(X, v1, v2, v3, v1_node, v2_node, v3_node, pow=1):
    I = X.astype(float) / 255.

    s1 = np.clip(1 - ((1. / ((v1_node - v1) ** 2)) * (v1 - I) ** 2) ** pow, 0, 1)
    s2 = np.clip(1 - ((1. / ((v2_node - v2) ** 2)) * (v2 - I) ** 2) ** pow, 0, 1)
    s3 = np.clip(1 - ((1. / ((v3_node - v3) ** 2)) * (v3 - I) ** 2) ** pow, 0, 1)

    return np.stack([s1, s2, s3], axis=2)


def optical_flow(current, prev):
    global of_hsv

    gray1 = cv2.GaussianBlur(current, dst=None, ksize=(3, 3), sigmaX=5)
    gray2 = cv2.GaussianBlur(prev, dst=None, ksize=(3, 3), sigmaX=5)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75,
                                        levels=3,
                                        winsize=5,
                                        iterations=3,
                                        poly_n=10,
                                        poly_sigma=1.2,
                                        flags=0)
    return flow


def flow_viz(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


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
        _frame_diff = frame_diff(prev, frame, blocksize=21)
        MHI1_f = MHI_1(MHI1_f, _frame_diff, tau=255, delta=20)
        #pseudocolor1 = MHI_pseudocolor(MHI1_f)
        #pseudocolor1b = MHI_pseudocolor2(MHI1_f)
        pseudocolor2 = MHI_springtime(MHI1_f)
        #pseudocolor3 = MHI_pseudocolor3(MHI1_f)
        #pseudocolor4 = MHI_pseudocolor4(MHI1_f)
        pseudocolor5 = pseudocolor_generator(MHI1_f, 1/3, 2/3, 1, 0, 1, 0.5, pow=0.5)
        pseudocolor6 = pseudocolor_generator(MHI1_f, 0.25, 0.5, 1, 0, 0.75, 0.5, pow=1)
        pseudocolor7 = pseudocolor_generator(MHI1_f, 0.25, 0.5, 1, 0, 1, 0.5, pow=1)


        # Display the frame
        cv2.imshow('frame_diff', _frame_diff)
        #cv2.imshow('MHI1', MHI1_f)
        #cv2.imshow("pseudocolor1", pseudocolor1)
        #cv2.imshow("pseudocolor1b", pseudocolor1b)
        #cv2.imshow("pseudocolor2", pseudocolor2)
        #cv2.imshow("pseudocolor3", pseudocolor3)
        #of = optical_flow(current=frame, prev=prev)
        #of_viz = flow_viz(of)

        #cv2.imshow("pseudocolor4", pseudocolor4)
        cv2.imshow("pseudocolor5", pseudocolor5)
        #cv2.imshow("pseudocolor6", pseudocolor6)
        #cv2.imshow("pseudocolor7", pseudocolor7)
        #cv2.imshow("opticalflow", of_viz)

    prev = frame

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
