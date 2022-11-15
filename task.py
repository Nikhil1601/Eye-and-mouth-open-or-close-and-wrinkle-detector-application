import cv2
import numpy as np

#creating facecascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# display(Image(filename=''))
#loading image to matrix
img = cv2.imread("WhatsApp Image 2022-09-16 at 6.36.49 PM (1).jpeg")

#converting into grayscale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=10)
for x,y,w,h in faces : 
    cropped_img = img[y:y+h,x:x+w]
    edges = cv2.Canny(cropped_img,130,1000)        
    number_of_edges = np.count_nonzero(edges)
cv2.imshow('img',4)
print(number_of_edges)
if number_of_edges > 1000:
    print("Wrinkle Found ")
else:
    print("No Wrinkle Found ")




