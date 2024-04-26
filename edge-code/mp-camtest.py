import cv2
import sys, time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
##false seems to give less performancec?
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize Pose with specific parameters
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Initialize Hands with specific parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_face_pose_hands_mesh(image):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image through the Face Mesh model
    face_results = face_mesh.process(rgb_image)
    # Process the image through the Pose model
    pose_results = pose.process(rgb_image)
    # Process the image through the Hands model
    hand_results = hands.process(rgb_image)

    annotated_image = image.copy()

    # Draw face mesh landmarks on the copied image if detected
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Draw pose landmarks on the copied image if detected
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
        )

    # Draw hand landmarks on the copied image if detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2)
            )

    return annotated_image
    

    
font = cv2.FONT_HERSHEY_SIMPLEX    
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False): 
  print("Unable to read camera feed")    
  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while cap.isOpened():
    s = time.time()
    ret, img = cap.read()  
    if ret == False:
        print('WebCAM Read Error')    
        sys.exit(0)
        
    annotated = get_face_pose_hands_mesh(img)
    e = time.time()
    fps = 1 / (e - s)
    cv2.putText(annotated, 'FPS:%5.2f'%(fps), (10,50), font, fontScale = 1,  color = (0,255,0), thickness = 1)
    cv2.imshow('webcam', annotated)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()