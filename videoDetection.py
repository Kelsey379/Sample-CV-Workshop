# import the library and define the cascade so that the program can use it to detect facial features
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#take input from the webcam
videoInput = cv2.VideoCapture(0)
#or, take input from a video:
#cap = cv2.VideoCapture('sampleVideo.mp4')

while True:
    #read the video frames
    _, img = videoInput.read()
    #convert the frame to a grayscae image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #use the Haar cascade to detect the faces

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #The detectMultiscale function takes 3 parameters: input image, scaleFactor and minNeighbours.
        # scaleFactor specifies how much the image size is reduced with each scale. 
        # minNeighbours specifies how many neighbors each candidate rectangle should have to retain it.
    
    #use a for loop to draw rectangles around any faces in the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #display the output:
    cv2.imshow('img', img)

    #if the user presses the esc key, the progrm will quit
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
    #release the video object
    videoInput.release