import cv2
haarcascade ="model/haarcascade_frontalface_default.xml"
capture=cv2.VideoCapture(0)
capture.set(3,640) #width
capture.set(4,480) #height
while True:
    success,img =capture.read()
    facecade = cv2.CascadeClassifier(haarcascade)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    face = facecade.detectMultiScale(img_gray,1.1,4)

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0,))

    cv2.imshow("Face",img)


    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break

