import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 


cap = cv.VideoCapture("video1.mp4")

pos = []
tempo = []
t = 0



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (800, 450))
    height, width, _ = frame.shape


    #Rastrear objetoc
    mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _,mask = cv.threshold(mask,100,255,cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(mask,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 600 and area < 10000:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),3)
            pos.append(x+w/2)
            tempo.append(t)
    t = t + 1

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break 

cap.release()
cv.destroyAllWindows()


def oha(t,a,b,w,p,c):
    return a*np.exp(-b*t)*np.cos(w*t - p) + c

plt.title('Grafico Pendulo')
plt.ylabel('Posicao X (pixels)')
plt.xlabel('Tempo (frames)')


guessA = (max(pos) - min(pos))/2
guess = [guessA,40,300,0,800]
tempo = np.array(tempo)
pos = np.array(pos)

values = curve_fit(oha, tempo, pos, guess)

a = values[0][0]
b = values[0][1]
w = values[0][2]
p = values[0][3]
c = values[0][4]

t = 2*np.pi / w 
q = 2*np.pi*(1/(1-np.exp(-2*b*t)))




plt.plot(tempo, oha(tempo,a,b,w,p,c) - c, color="red")
plt.plot(tempo, pos - c, "bo", markersize = 2, color = 'blue')
print(q)
plt.show()



