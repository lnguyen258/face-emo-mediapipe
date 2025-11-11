import cv2
import mediapipe as mp
import csv
import os
import joblib  
import numpy as np  

def load_label_names(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

landmark_csv_path = 'face_landmarks.csv'
label_csv_path = 'face_labels.csv'

save_mode = False
eval_mode = False 

# LOAD THE TRAINED MODEL
model_path = 'face_emotion_mlp.joblib'  
model = joblib.load(model_path)  
label_names = load_label_names(label_csv_path)  

with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

            face_landmarks = results.multi_face_landmarks[0]
            landmarks_coords = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                landmarks_coords.extend([x, y])
        else:
            landmarks_coords = []

        if eval_mode and landmarks_coords:
            input_array = np.array(landmarks_coords).reshape(1, -1)
            pred_label = model.predict(input_array)[0]
            pred_name = label_names[int(pred_label)] if int(pred_label) < len(label_names) else "Unknown"
            cv2.putText(frame, f'Emotion: {pred_name}', (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f'Faces detected: {len(results.multi_face_landmarks) if results.multi_face_landmarks else 0}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mode_text = 'ON' if save_mode else 'OFF'
        cv2.putText(frame, f'Save mode: {mode_text} (Press S to toggle)',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        eval_text = 'ON' if eval_mode else 'OFF'  
        cv2.putText(frame, f'Eval mode: {eval_text} (Press E to toggle)', 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'Press 0-9 to save label with landmarks',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'Press Q to quit',
                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Landmark Tracking', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_mode = not save_mode
            print(f'Save mode {"activated" if save_mode else "deactivated"}')
        elif key == ord('e'):  
            eval_mode = not eval_mode
            print(f'Evaluation mode {"activated" if eval_mode else "deactivated"}')
        elif save_mode and key in [ord(str(d)) for d in range(10)]:
            label = chr(key)
            if landmarks_coords:
                file_exists_landmark = os.path.isfile(landmark_csv_path)
                with open(landmark_csv_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    if not file_exists_landmark:
                        header = ['label']
                        num_points = len(landmarks_coords) // 2
                        for i in range(num_points):
                            header.append(f'x{i}')
                            header.append(f'y{i}')
                        writer.writerow(header)
                    writer.writerow([label] + landmarks_coords)

                print(f'Saved landmarks with label {label} to {landmark_csv_path}.')

# Release resources
cap.release()
cv2.destroyAllWindows()
