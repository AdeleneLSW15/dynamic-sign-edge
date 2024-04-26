import cv2
import sys, time
import mediapipe as mp
import numpy as np
import tensorflow as tf
import nanocamera as nano


# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Setup MediaPipe instances
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_face_pose_hands_mesh(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image through the MediaPipe models
    face_results = face_mesh.process(rgb_image)
    pose_results = pose.process(rgb_image)
    hand_results = hands.process(rgb_image)

    annotated_image = image.copy()
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Draw detected landmarks on the image for the face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
        )

    # Draw hand landmarks and classify left or right
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            color = (245, 117, 66) if handedness.classification[0].label == 'Right' else (121, 44, 250)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2)
            )

    keypoints = extract_keypoints(face_results, pose_results, hand_results)
    return annotated_image, keypoints

def extract_keypoints(face_results, pose_results, hand_results):
    # Flatten the coordinates of the detected landmarks from each results object
    pose = np.array([[res.x, res.y, res.z] for res in pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33*3)
    
    # Correctly extract face landmarks
    if face_results.multi_face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for face_landmarks in face_results.multi_face_landmarks for lm in face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)
    
    # Initialize arrays for left and right hands
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            if handedness.classification[0].label == 'Left':
                lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            elif handedness.classification[0].label == 'Right':
                rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, face, lh, rh])

#Visualize Probability on imshow!
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


#Load TFLite model
model_path = "AK-GRU16-LR-Dense256-gelu-fs=15-cls=3.tf"
# Load the model
loaded_model = tf.keras.models.load_model(model_path)


#main application code
sequence = []
sentence = []
threshold = 0.7
new_width = 680
new_height = 480
actions = ['listen', 'look', 'shhh']
trained_frame = 60

#for FPS print
fps_text_x = new_width - 10  # 10 pixels from the right edge
fps_text_y = new_height - 10  # 10 pixels from the bottom edge

font = cv2.FONT_HERSHEY_SIMPLEX    
camera = nano.Camera(flip = 0, width = 640, height = 480, fps = 25)
if (cap.isOpened() == False): 
  print("Unable to read camera feed")    
  
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('CSI Camera ready? - ', camera.isReady())
while camera.isReady():
  try: 
    s = time.time()
    img = camera.read()  
    resized_frame = cv2.resize(img, (640, 480))

    #Make detections
    image, keypoints = get_face_pose_hands_mesh(img)
    sequence.append(keypoints)
    sequence = sequence[-trained_frame:]
    
    if len(sequence) == trained_frame:
        res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        #visualization logic
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) > 5:
                sentence = sentence[-5:]
        #Visualize probabilities
        image = prob_viz(res, actions, image, colors)
        
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    e = time.time()
    fps = 1 / (e - s)
    print('FPS:%5.2f'%(fps))
    #cv2.putText(image, 'FPS:%5.2f'%(fps), (fps_text_x,fps_text_y), font, fontScale = 1,  color = (0,255,0), thickness = 1)
    cv2.imshow('webcam', img)
    cv2.imshow('Sign Recognition Feed', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  except KeyboardInterrupt:
      break

#close camera instance
camera.release()
#remove camera object
del camera