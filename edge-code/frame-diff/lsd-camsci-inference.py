#display on opencv2 using nanocamera module for CSI camera
#Then live frame difference
import cv2
import nanocamera as nano
import numpy as np
from math import pi


#frame difference algorithm
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

#get the second frame difference
def frame_diff2(prev, curr, kernel=np.array((5, 5), dtype=np.uint8),
                filter_size=3, filter_size_post=3, blocksize=15, C=3):
    frame_diff = cv2.subtract(curr, prev)
    frame_diff = cv2.medianBlur(frame_diff, filter_size)
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blocksize, C)
    mask = cv2.medianBlur(mask, filter_size_post)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

#obtain first MHI
def MHI_1(prev_mhi, frame_diff, tau=255, delta=32):
    next_MHI = np.zeros_like(prev_mhi).astype(int)

    next_MHI[frame_diff == 255] = tau
    next_MHI[frame_diff != 255] = np.clip(prev_mhi[frame_diff != 255].astype(int) - delta, 0, 255)

    return np.clip(next_MHI, 0, 255).astype(np.uint8)

#psuedocolor implementation
def pseudocolor_generator(X, v1, v2, v3, v1_node, v2_node, v3_node, pow=1):
    I = X.astype(float) / 255.

    s1 = np.clip(1 - ((1. / ((v1_node - v1) ** 2)) * (v1 - I) ** 2) ** pow, 0, 1)
    s2 = np.clip(1 - ((1. / ((v2_node - v2) ** 2)) * (v2 - I) ** 2) ** pow, 0, 1)
    s3 = np.clip(1 - ((1. / ((v3_node - v3) ** 2)) * (v3 - I) ** 2) ** pow, 0, 1)

    return np.stack([s1, s2, s3], axis=2)

#Load TFLite model
model_path = "AK-Transformer-e=64-h=2-ff=128-D=0.1-Dense=128.tf"
# Load the model
loaded_model = tf.keras.models.load_model(model_path)


#Live implementaiton - camera code
camera = nano.Camera(flip = 0, width = 640, height = 480, fps = 25)
print('CSI Camera ready? - ', camera.isReady())

prev = None
MHI1_f = None
of_hsv = None

while camera.isReady():
    try:
        #read camera image
        frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if MHI1_f is None and prev is not None:
            MHI1_f = np.zeros_like(prev)

        if of_hsv is None:
            of_hsv = np.zeros_like(frame)


        if prev is not None:
            _frame_diff = frame_diff(prev, frame)
            MHI1_f = MHI_1(MHI1_f, _frame_diff, tau=255, delta=32)
            pseudocolor5 = pseudocolor_generator(MHI1_f, 1/3, 2/3, 1, 0, 1, 0.5, pow=1)

            #display the frame
            cv2.imshow("psuedocolor5", pseudocolor5)
        
        prev = frame

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
         break

#close camera instance
camera.release()
#remove camera object
del camera 
