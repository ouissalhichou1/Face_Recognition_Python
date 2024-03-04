from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
import face_recognition
import os

def btn_clicked():
    print("Button Clicked")

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def recognize_faces_in_image():
    img_path = filedialog.askopenfilename()
    if img_path:
        img = cv2.imread(img_path)
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


def recognize_faces_in_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)

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


def recognize_live_camera():
    cap = cv2.VideoCapture(0)  # Initialize the VideoCapture object

    while cap.isOpened():
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
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_rectangle(img, face_loc):
    top, right, bottom, left = face_loc
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

# Function to draw text indicating the name of the recognized person
def draw_text(img, name, face_loc):
    top, right, bottom, left = face_loc
    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)


window = Tk()

window.geometry("720x480")
window.configure(bg="#000000")

canvas = Canvas(
    window,
    bg="#000000",
    height=480,
    width=720,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(file="background.png")
background = canvas.create_image(
    95.0, 204.0,
    image=background_img)

img0 = PhotoImage(file="img0.png")
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=recognize_faces_in_image,
    relief="flat")

b0.place(
    x=338, y=373,
    width=288,
    height=37)

img1 = PhotoImage(file="img1.png")
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=recognize_live_camera,
    relief="flat")

b1.place(
    x=337, y=239,
    width=291,
    height=37)

img2 = PhotoImage(file="img2.png")
b2 = Button(
    image=img2,
    borderwidth=0,
    highlightthickness=0,
    command=recognize_faces_in_video,
    relief="flat")

b2.place(
    x=338, y=306,
    width=291,
    height=37)

canvas.create_text(
    484.0, 128.0,
    text="Welcome to the FUTURE",
    fill="#ffffff",
    font=("Inter-Bold", int(24.0)))

path = 'persons'
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
class_names = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
encode_list_known = find_encodings(images)

window.resizable(False, False)
window.mainloop()
