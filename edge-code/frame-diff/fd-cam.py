import cv2

cap  = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret, prev_frame = cap.read()
ret, current_frame = cap.read()
ret, next_frame = cap.read()

while True:
    # Frame differencing technique
    motion_diff = frame_diff(prev_frame, current_frame, next_frame)

    if motion_diff is not None:
        # Convert to grayscale for thresholding
        motion_diff_grayscale = cv2.cvtColor(motion_diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(motion_diff_grayscale, 30, 255, cv2.THRESH_BINARY)

        # Display the result
        cv2.imshow("Motion Diff Frame", thresh)

    # Update frames
    prev_frame = current_frame
    current_frame = next_frame
    ret, next_frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting ...")
        break

    # Break loop on 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
