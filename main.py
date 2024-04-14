import cv2
import numpy as np
from keras.models import load_model

age_prediction_model = load_model("model/age_prediction_model.keras")
head_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


index_to_age = {0: "0 - 5",
                1: "5 - 10",
                2: "10 - 20",
                3: "20 - 40",
                4: "40 - 60",
                5: "60 - 80",
                6: "80 - 100",
                7: "100 - 120"}

def detect_heads(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heads = head_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    head_regions = []
    for (x, y, w, h) in heads:
        head_regions.append((x, y, w, h))
    return head_regions

def age_prediction_to_num(prediction_array):
    nums_array = np.array(prediction_array)
    return np.argmax(nums_array)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    head_regions = detect_heads(frame)

    for (x, y, w, h) in head_regions:
        head_region = frame[y:y+h, x:x+w]
        head_region_resized = cv2.resize(head_region, (48, 48))
        head_region_gray = cv2.cvtColor(head_region_resized, cv2.COLOR_BGR2GRAY)

        # Preprocess the image for input into the age prediction model
        head_region_gray = np.expand_dims(head_region_gray, axis=0)  # Add batch dimension
        head_region_gray = np.expand_dims(head_region_gray, axis=-1)  # Add channel dimension

        # Perform age prediction
        predicted_age = age_prediction_model.predict(head_region_gray)

        age_index = age_prediction_to_num(predicted_age)
        age = index_to_age[age_index]

        # Draw bounding box around detected head
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display predicted age label on the frame
        cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
