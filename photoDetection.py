import cv2
# Load the cascade (For this workshop we're using a premade Haar cascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('people.jpeg')
# Convert the image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect the faces using the face cascade
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output (change to save image results later)
cv2.imshow('img', img)
cv2.waitKey(0)