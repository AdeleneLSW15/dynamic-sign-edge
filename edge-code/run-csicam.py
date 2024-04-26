#display on opencv2 using nanocamera module for CSI camera
import cv2
import nanocamera as nano

camera = nano.Camera(flip = 0, width = 640, height = 480, fps = 25)
print('CSI Camera ready? - ', camera.isReady())
while camera.isReady():
    try:
         #read camera image
         frame = camera.read()
         #display the frame
         cv2.imshow("Video Frame", frame)
         if cv2.waitKey(25) & 0xFF == ord('q'):
             break
    except KeyboardInterrupt:
         break
#close camera instance
camera.release()
#remove camera object
del camera 
