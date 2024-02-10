import cv2
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog


# Function to find encodings of known images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


# Function to recognize faces in an image
def recognize_faces_in_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img_rgb)
    encode_faces = face_recognition.face_encodings(img_rgb, face_locations)

    for encode_face, face_loc in zip(encode_faces, face_locations):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distances = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = class_names[match_index].upper()
            draw_rectangle(img, face_loc)
            draw_text(img, name, face_loc)

    cv2.imshow('Face Recognition - Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to recognize faces in a video
def recognize_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)
        encode_faces = face_recognition.face_encodings(frame_rgb, face_locations)

        for encode_face, face_loc in zip(encode_faces, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_distances = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = class_names[match_index].upper()
                draw_rectangle(frame, face_loc)
                draw_text(frame, name, face_loc)

        cv2.imshow('Face Recognition - Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to draw a rectangle around detected faces
def draw_rectangle(img, face_loc):
    top, right, bottom, left = face_loc
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)


# Function to draw text indicating the name of the recognized person
def draw_text(img, name, face_loc):
    top, right, bottom, left = face_loc
    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)


# Create Tkinter root window
root = tk.Tk()
root.title("Recognition App")

# Set the geometry of the root window
root.geometry("400x200")  # Width x Height
root.configure(bg="#B3BABC")  # Set background color

# Create a label for the title
title_label = tk.Label(root, text="Welcome to the Future", font=("Helvetica", 16), bg="#B3BABC")
title_label.pack(pady=10)  # Add padding

# Create a frame to hold the buttons
button_frame = tk.Frame(root, bg="#B3BABC")
button_frame.pack(expand=True)

# Function to handle live camera recognition
# Function to handle live camera recognition
# Function to handle live camera recognition
def recognize_live_camera():
    cap = cv2.VideoCapture(0)  # Initialize the VideoCapture object

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(frame_rgb)
            encode_faces = face_recognition.face_encodings(frame_rgb, face_locations)

            for encode_face, face_loc in zip(encode_faces, face_locations):
                matches = face_recognition.compare_faces(encode_list_known, encode_face)
                face_distances = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_distances)

                if matches[match_index]:
                    name = class_names[match_index].upper()
                else:
                    name = "Unknown"

                draw_rectangle(frame, face_loc)
                draw_text(frame, name, face_loc)

            cv2.imshow('Live Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        else:
            root.after(10, update_frame)  # Call update_frame again after 10 milliseconds

    update_frame()  # Start the update_frame function



# Function to handle video recognition
def recognize_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_faces_in_video(file_path)


# Function to handle image recognition
def recognize_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_faces_in_image(file_path)


# Create buttons for different recognition options
live_camera_button = tk.Button(button_frame, text="Recognize Live Camera", command=recognize_live_camera,
                                bg="#C3CED2", width=20)
live_camera_button.pack(side="top", pady=5)

video_button = tk.Button(button_frame, text="Recognize in Video", command=recognize_video,
                            bg="#C3CED2", width=20)
video_button.pack(side="top", pady=5)

image_button = tk.Button(button_frame, text="Recognize in Picture", command=recognize_image,
                            bg="#C3CED2", width=20)
image_button.pack(side="top", pady=5)

# Load known images and encodings
path = 'persons'
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
class_names = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
encode_list_known = find_encodings(images)

root.mainloop()
