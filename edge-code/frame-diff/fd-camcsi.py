#display on opencv2 using nanocamera module for CSI camera
#Then live frame difference
import cv2
import nanocamera as nano


#frame difference algorithm
def frame_diff(prev_frame, current_frame, next_frame):
    if prev_frame is None or current_frame is None or next_frame is None:
        return None
    #get absolute difference between frames
    diff1 = cv2.absdiff(next_frame,current_frame)
    diff2 = cv2.absdiff(current_frame,prev_frame)
    motion_diff = cv2.bitwise_and(diff1,diff2)

    return motion_diff

camera = nano.Camera(flip = 0, width = 640, height = 480, fps = 25)
print('CSI Camera ready? - ', camera.isReady())


prev_frame = camera.read()
current_frame = camera.read()
next_frame = camera.read()

while camera.isReady():
    try:
         #read camera image
         motion_diff = frame_diff(prev_frame,current_frame,next_frame)
         if motion_diff is not None:
             motion_diff_grayscale = cv2.cvtColor(motion_diff,cv2.COLOR_BGR2GRAY)
             #motion difference threshold
             _, thresh = cv2.threshold(motion_diff_grayscale, 30, 255, cv2.THRESH_BINARY)
             cv2.imshow("Difference Frame", thresh)
         #read next frames
         prev_frame = current_frame
         current_frame = next_frame
         next_frame = camera.read()

         if cv2.waitKey(25) & 0xFF == ord('q'):
             break
    except KeyboardInterrupt:
         break

#close camera instance
camera.release()
#remove camera object
del camera 
