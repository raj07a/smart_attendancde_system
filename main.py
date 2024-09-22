import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import streamlit as st

# Paths
IMAGE_FOLDER = 'D:/repository/Attendance System/train_img/'
ATTENDANCE_FILE = 'attendance.csv'

# Function to load images and generate encodings from subfolders
def load_images_from_folder(folder):
    known_faces = []
    known_names = []
    known_enrollments = []
    
    for student_folder in os.listdir(folder):
        student_path = os.path.join(folder, student_folder)
        if os.path.isdir(student_path):
            try:
                name, enrollment = student_folder.split('_')
            except ValueError:
                print(f"Invalid folder name format for: {student_folder}")
                continue  # Skip folders that don't follow the convention
            
            for image_filename in os.listdir(student_path):
                image_path = os.path.join(student_path, image_filename)
                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(name)
                    known_enrollments.append(enrollment)
                else:
                    print(f"Face not detected in {image_filename} for {name}")
    
    return known_faces, known_names, known_enrollments

# Function to mark attendance
def mark_attendance(name, enrollment):
    now = datetime.now()
    time_string = now.strftime('%Y-%m-%d %H:%M:%S')

    try:
        attendance = pd.read_csv(ATTENDANCE_FILE)
    except FileNotFoundError:
        attendance = pd.DataFrame(columns=['Name', 'Enrollment', 'Time'])

    # Only mark attendance if the student hasn't been marked present already
    if enrollment not in attendance['Enrollment'].values:
        new_entry = pd.DataFrame([{'Name': name, 'Enrollment': enrollment, 'Time': time_string}])
        attendance = pd.concat([attendance, new_entry], ignore_index=True)
        attendance.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False

# Initialize known faces
known_face_encodings, known_face_names, known_enrollment_numbers = load_images_from_folder(IMAGE_FOLDER)

# Real-time face recognition logic with a recognition buffer
def recognize_faces():
    recognized_students = {}
    present_students = []
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            enrollment = None

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                enrollment = known_enrollment_numbers[match_index]

                if enrollment not in recognized_students:
                    recognized_students[enrollment] = datetime.now()
                else:
                    if (datetime.now() - recognized_students[enrollment]).total_seconds() > 3:
                        if mark_attendance(name, enrollment):
                            present_students.append((name, enrollment))
                        del recognized_students[enrollment]

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return present_students

# Streamlit UI
def streamlit_ui():
    st.title("Face Recognition Attendance System Dashboard")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    start_recognition = st.sidebar.button("Start Face Recognition")

    if start_recognition:
        present_students = recognize_faces()
        st.write("### Students Present in this Session:")
        for name, enrollment in present_students:
            st.success(f"{name} (Enrollment: {enrollment})")

    # Display Attendance History
    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
        st.write("### Attendance History")
        st.dataframe(attendance_df)
    else:
        st.write("No attendance history available.")

# Run Streamlit app
if __name__ == "__main__":
    streamlit_ui()
