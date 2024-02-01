import cv2
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding Complete.')

def recognize_faces_in_image(image_path):
    img = cv2.imread(image_path)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(imgS)
    encodeFaces = face_recognition.face_encodings(imgS, faceLocations)

    for encodeFace, faceLoc in zip(encodeFaces, faceLocations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistances = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDistances)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"The person in the image is: {name}")

def open_camera():
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurentFrame = face_recognition.face_locations(imgS)
        encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

        for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeface)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_faces_in_image(file_path)

def main():
    print("Choose an option:")
    print("1. Select a picture")
    print("2. Open camera")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        select_image()
    elif choice == '2':
        open_camera()
    else:
        print("Invalid choice. Please enter '1' or '2'.")

if __name__ == "__main__":
    main()
