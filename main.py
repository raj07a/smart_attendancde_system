import cv2
import face_recognition
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Load the attendance data
attendance_file = 'attendance.csv'

# Function to load attendance data
def load_attendance():
    try:
        return pd.read_csv(attendance_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=['Enrollment', 'Name', 'Time'])

# Function to mark attendance if a face is recognized for 3 seconds
def mark_attendance(enrollment, name):
    attendance = load_attendance()
    now = datetime.now()
    time_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if not ((attendance['Enrollment'] == enrollment) & (attendance['Name'] == name)).any():
        new_entry = pd.DataFrame([{'Enrollment': enrollment, 'Name': name, 'Time': time_string}])
        attendance = pd.concat([attendance, new_entry], ignore_index=True)
        attendance.to_csv(attendance_file, index=False)
        st.success(f"Attendance marked for {name} ({enrollment}) at {time_string}")

# Function to load images and generate encodings
def load_images_from_folder(folder):
    known_faces = []
    known_names = []
    known_enrollments = []

    for student_folder in os.listdir(folder):
        student_path = os.path.join(folder, student_folder)
        if os.path.isdir(student_path):
            enrollment, name = student_folder.split('_')
            for filename in os.listdir(student_path):
                img_path = os.path.join(student_path, filename)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(name)
                    known_enrollments.append(enrollment)

    return known_faces, known_names, known_enrollments

# Load known faces and their corresponding names and enrollments
IMAGE_FOLDER = 'train_img/'  # Adjust this path as needed
known_face_encodings, known_face_names, known_face_enrollments = load_images_from_folder(IMAGE_FOLDER)

# Real-time face recognition function
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    face_detection_count = {}

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            enrollment = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                enrollment = known_face_enrollments[match_index]

                if enrollment not in face_detection_count:
                    face_detection_count[enrollment] = 1
                else:
                    face_detection_count[enrollment] += 1

                if face_detection_count[enrollment] >= 90:  # Roughly 3 seconds
                    mark_attendance(enrollment, name)
                    face_detection_count[enrollment] = 0  # Reset count after marking attendance

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({enrollment})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        st.image(frame, channels="BGR")
        if st.button('Stop Recognition'):
            break

    video_capture.release()

# Streamlit UI layout
st.title("Student Attendance System")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose the action:", ["Real-time Attendance", "View Attendance History"])

if option == "Real-time Attendance":
    st.header("Real-time Face Recognition for Attendance")
    st.write("Press the button below to start face recognition and mark attendance.")
    if st.button("Start Face Recognition"):
        recognize_faces()

elif option == "View Attendance History":
    st.header("Attendance History")
    attendance = load_attendance()
    st.write(attendance)

    if st.button("Clear Attendance History"):
        attendance = pd.DataFrame(columns=['Enrollment', 'Name', 'Time'])
        attendance.to_csv(attendance_file, index=False)
        st.success("Attendance history cleared.")
