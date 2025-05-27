# Face-recognition-
!pip install face_recognition opencv-python numpy


import cv2
import numpy as np
import os
from google.colab import files
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_face():
    data = next(iter(files.upload().values()))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w], img, faces

print("Upload your image")
ref_face, _, _ = get_face()

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.train([ref_face], np.array([0]))

# 3. Upload a different face to test similarity
print("Upload a different face")
test_face, test_img, faces = get_face()

label, confidence = recogniser.predict(test_face)
print(f"Confidence: {confidence:.1f}", "✅ same person" if confidence < 50 else "
