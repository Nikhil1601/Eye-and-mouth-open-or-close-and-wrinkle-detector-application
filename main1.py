
import numpy as np
import cv2
from scipy.spatial import distance as dst
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import math
top =tk.Tk()
top.geometry('800x600')
top.title()
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

harcascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(harcascade)

def eye(a,b,c,d,e,f):
    x=dst.euclidean(b,d)
    y=dst.euclidean(c,e)
    z=dst.euclidean(a,f)
    ratio=(x+y)/(2.0*z)

    if ratio>0.16:
        return 1
    else :return 0


def mouth(x,y):
    x_mean=np.mean(x,axis=0)
    y_mean=np.mean(y,axis=0)
    
    return abs(x_mean-y_mean)




def wrinkles(image):
    frame = image
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=10)
    for x,y,w,h in faces : 
        frame = frame[y:y+h,x:x+w]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    numbers_of_edges=np.count_nonzero(edges)
    print(numbers_of_edges)
    if numbers_of_edges>3500:
        return 1
    else: return 0


def Detect(file_path):
    img = cv2.imread(file_path)
    wrinlke=wrinkles(img,)
    img = cv2.resize(img,(512,512))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(img_gray)

    LBFmodel = 'lbfmodel.yaml'
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _,landmarks = landmark_detector.fit(img_gray,faces)

    n=0
    l=[]
    for landmark in landmarks:
        for x,y in landmark[0]:
            n=n+1
            cv2.circle(img,(int(x),int(y)),2,(0,255,0),2)
            cv2.putText(img,str(n),(int(x),int(y-8)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
            l.append(int(x))
            l.append(int(y))
    left_eye=eye(l[37],l[38],l[39],l[42],l[41],l[40])
    right_eye=eye(l[43],l[44],l[45],l[48],l[47],l[46])


    top_lip=[]
    bottom_lip=[]
    for x in range(50,55):
        top_lip.append(l[x])
    for x in range(50,55):
        top_lip.append(l[x])
    for x in range(62,65):
        bottom_lip.append(l[x])
    for x in range(66,69):
        bottom_lip.append(l[x])    
    mouth_cal=mouth(top_lip,bottom_lip)
    print(mouth_cal)
    # if mouth_cal>9.75:
    #     label1.configure(foreground="#011638",text = 'mouth open')
    # else: return label1.configure(foreground="#011638",text = 'mouth close')


    if (left_eye==0 or right_eye==0) and (mouth_cal>9) and (wrinlke==1):
        label1.configure(foreground="#011638",text = 'eyes closed \n mouth open \n wrinkle found')
    elif (left_eye==0 or right_eye==0) and (mouth_cal>9) and (wrinlke!=1):
        label1.configure(foreground="#011638",text = 'eye closed \n mouth open \n wrinkle not found')
    elif (left_eye==0 or right_eye==0) and (mouth_cal<9) and (wrinlke==1):
        label1.configure(foreground="#011638",text = 'eye closed \n mouth closed \n wrinkle found')
    elif (left_eye==0 or right_eye==0) and (mouth_cal<9) and (wrinlke!=1):
        label1.configure(foreground="#011638",text = 'eye closed \n mouth closed \n wrinkle not found')
    elif (left_eye!=0 or right_eye!=0) and (mouth_cal>9) and (wrinlke==1):
        label1.configure(foreground="#011638",text = 'eyes open \n mouth open \n wrinkle found')
    elif (left_eye!=0 or right_eye!=0) and (mouth_cal>9) and( wrinlke!=1):
        label1.configure(foreground="#011638",text = 'eye open \n mouth open \n wrinkle not found')
    elif (left_eye!=0 or right_eye!=0) and (mouth_cal<9) and (wrinlke==1):
        label1.configure(foreground="#011638",text = 'eye open \n mouth closed \n wrinkle found')
    else:
        label1.configure(foreground="#011638",text = 'eye open \n mouth closed \n wrinkle not found')
    # cv2.imshow('original', img)
def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        img = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=img)
        sign_image.image = img
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='EYE,MOUTH And Wrinkles',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()

