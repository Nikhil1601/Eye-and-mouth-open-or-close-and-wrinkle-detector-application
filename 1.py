import numbers
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# display(Image(filename=''))
while True:
    frame = cv2.imread('WhatsApp Image 2022-09-16 at 6.36.50 PM.jpeg')
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=10)
    for x,y,w,h in faces : 
        frame = frame[y:y+h,x:x+w]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    numbers_of_edges=np.count_nonzero(edges)
    print(numbers_of_edges)
    # cv2.imshow("edges", edges)
    # cv2.imshow("gray", gray)
    # if cv2.waitKey(1) == ord("q"):
    #     break


cv2.waitKey(0)
cv2.destroyAllWindows()